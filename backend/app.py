"""FastAPI WebSocket server that opens the webcam on client connect and
processes frames with MediaPipe Hands.

Usage:
    uvicorn app:app --reload

Behavior (per-connection):
- Camera is opened when a client connects to /ws (not at server start)
- MediaPipe Hands is initialized with the requested parameters
- For each frame: BGR->RGB, pass to MediaPipe, if hands detected send a
  summarized JSON message (no full landmark dump)
- Messages are rate-limited to 5/s (min interval 0.2s)
- Camera and MediaPipe are released on disconnect or error

Constraints satisfied: no TTS, no recognition, no storage, single file
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import mediapipe as mp
import os
from collections import deque
from typing import List, Tuple

# Model / inference configuration
SEQ_LEN = 30  # number of frames in sequence for model input (configurable)
CONFIDENCE_THRESHOLD = 0.7  # minimum confidence to accept model prediction
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sign_model.pt")

# Robust import for MediaPipe Hands.
MPHands = None
# Tasks-mode hand landmarker (MediaPipe Tasks API) support
TASK_HAND_LANDMARKER = None
TASK_IMAGE_MODULE = None
_TASK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
try:
    # Preferred import path
    from mediapipe.solutions.hands import Hands as MPHands  # type: ignore
except Exception:
    # Fallback: try importing mediapipe and accessing solutions safely
    try:
        import mediapipe as _mediapipe  # type: ignore
        sols = getattr(_mediapipe, "solutions", None)
        if sols is not None:
            hands_mod = getattr(sols, "hands", None)
            if hands_mod is not None and hasattr(hands_mod, "Hands"):
                MPHands = hands_mod.Hands
        # Additional fallback: some mediapipe builds expose the "solutions"
        # package under a nested `mediapipe.python` namespace (or similar).
        # Try the common alternative path before giving up.
        if MPHands is None:
            try:
                # e.g. mediapipe.python.solutions.hands
                import importlib
                alt = importlib.import_module("mediapipe.python.solutions.hands")
                if hasattr(alt, "Hands"):
                    MPHands = getattr(alt, "Hands")
            except Exception:
                # ignore and continue to outer fallback
                pass
    except Exception:
        # If import fails, leave MPHands as None and handle at runtime
        MPHands = None
        logging.warning(
            "Unable to import MediaPipe Hands; hand detection will be disabled."
        )

# Helper: try to initialize the MediaPipe Tasks HandLandmarker when available
def _ensure_tasks_hand_landmarker() -> Optional[object]:
    """Return a HandLandmarker instance (Tasks API) or None.

    Will attempt to import the Tasks hand landmarker module and load a
    model from `backend/models/hand_landmarker.task`. If the model is not
    present, attempts to download it from a best-effort list of URLs. If
    anything fails, returns None and the caller should fall back to
    heartbeat-only mode.
    """

    global TASK_HAND_LANDMARKER, TASK_IMAGE_MODULE, _TASK_MODEL_PATH

    if TASK_HAND_LANDMARKER is not None:
        return TASK_HAND_LANDMARKER

    try:
        # import inside function to avoid hard dependency at module import
        from mediapipe.tasks.python.vision import hand_landmarker as hl
        from mediapipe.tasks.python.vision import core as vision_core
        from mediapipe.tasks.python.vision.core import image as image_lib
        from mediapipe.tasks.python.vision.core import vision_task_running_mode

        TASK_IMAGE_MODULE = image_lib

        # ensure models dir exists
        models_dir = os.path.dirname(_TASK_MODEL_PATH)
        os.makedirs(models_dir, exist_ok=True)

        # If model file not present, try to download from a couple of well-known locations
        if not os.path.exists(_TASK_MODEL_PATH):
            candidates = [
                "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task",
            ]
            for url in candidates:
                try:
                    logging.info("Attempting to download MediaPipe hand_landmarker model from %s", url)
                    import urllib.request

                    urllib.request.urlretrieve(url, _TASK_MODEL_PATH)
                    logging.info("Downloaded hand_landmarker model to %s", _TASK_MODEL_PATH)
                    break
                except Exception:
                    logging.debug("Failed to download model from %s", url, exc_info=True)

        if not os.path.exists(_TASK_MODEL_PATH):
            logging.warning(
                "MediaPipe Tasks model not found at %s; Tasks HandLandmarker will not be available",
                _TASK_MODEL_PATH,
            )
            return None

        # create the HandLandmarker instance in VIDEO running mode so that
        # detect_for_video can be used. create_from_model_path defaults to
        # IMAGE mode which causes `Task is not initialized with the video mode`.
        try:
            # Build options with VIDEO running mode. Use the HandLandmarker
            # module's BaseOptions helper to point to our model file.
            options = hl.HandLandmarkerOptions(
                base_options=hl._BaseOptions(model_asset_path=_TASK_MODEL_PATH),
                running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
                num_hands=2,
            )
            TASK_HAND_LANDMARKER = hl.HandLandmarker.create_from_options(options)
            logging.info("Initialized MediaPipe Tasks HandLandmarker (VIDEO mode) from %s", _TASK_MODEL_PATH)
            return TASK_HAND_LANDMARKER
        except Exception:
            # Fallback: try the simple factory which creates an IMAGE-mode
            # hand landmarker; we'll log and return it (but detect_for_video
            # will not work in IMAGE mode).
            logging.exception("Failed to initialize HandLandmarker in VIDEO mode; falling back to IMAGE mode")
            TASK_HAND_LANDMARKER = hl.HandLandmarker.create_from_model_path(_TASK_MODEL_PATH)
            logging.info("Initialized MediaPipe Tasks HandLandmarker (IMAGE mode) from %s", _TASK_MODEL_PATH)
            return TASK_HAND_LANDMARKER
    except Exception:
        logging.exception("Failed to initialize MediaPipe Tasks HandLandmarker")
        TASK_HAND_LANDMARKER = None
        TASK_IMAGE_MODULE = None
        return None
import threading
import pyttsx3

# Global toggle to enable/disable TTS easily
TTS_ENABLED = True

# Streaming stability configuration (tweakable)
# Minimum number of occurrences required to validate a sign within the time window
STABILITY_N = 5
# Time window (seconds) in which STABILITY_N occurrences must appear
STABILITY_WINDOW = 2.0
# Minimum cooldown (seconds) before re-emitting the same validated text
REPETITION_COOLDOWN = 3.0
# Maximum length of per-frame text buffer
BUFFER_MAXLEN = 15
# Confidence to attach to simulated recognizer outputs
DEFAULT_TEXT_CONFIDENCE = 0.9

# Initialize pyttsx3 engine at module import (startup). We choose a French
# voice if available. This is done once and reused by speak_async.
_tts_engine = None
_tts_lock = threading.Lock()
_tts_current = {"text": None, "thread": None}

try:
    _tts_engine = pyttsx3.init()
    # try to select a French voice if available (best-effort)
    voices = _tts_engine.getProperty("voices") or []
    selected = None
    for v in voices:
        name = (getattr(v, "name", "") or "").lower()
        vid = (getattr(v, "id", "") or "").lower()
        if "fr" in name or "french" in name or "fr" in vid:
            selected = v.id
            break
    if selected:
        try:
            _tts_engine.setProperty("voice", selected)
        except Exception:
            pass
    try:
        _tts_engine.setProperty("rate", 150)
    except Exception:
        pass
except Exception:
    logging.exception("Failed to initialize pyttsx3 TTS engine — TTS will be disabled")
    _tts_engine = None


def speak_async(text: str) -> None:
    """Non-blocking TTS: speak text in a daemon thread.

    - Ignores request if the same text is already being played.
    - Uses a lock around pyttsx3 engine because it is not thread-safe.
    - Exceptions are caught and logged to avoid crashing the server.
    """

    if not TTS_ENABLED:
        return
    if not text:
        return
    if _tts_engine is None:
        logging.warning("TTS engine not available; skipping speak_async")
        return

    # Prevent duplicate playback of the same text
    with _tts_lock:
        current_text = _tts_current.get("text")
        current_thread = _tts_current.get("thread")
        if current_text == text and current_thread is not None and current_thread.is_alive():
            logging.debug("Same text is already being played; skipping TTS")
            return

        # Start a new daemon thread to run TTS
        def _worker(s: str):
            try:
                with _tts_lock:
                    # set the marker so reentrant calls can detect active playback
                    _tts_current["text"] = s
                # do the actual speaking without holding the lock to allow
                # checks from other threads (pyttsx3 itself needs serialized access)
                try:
                    _tts_engine.say(s)
                    _tts_engine.runAndWait()
                except Exception:
                    logging.exception("Error during pyttsx3 speak")
            finally:
                with _tts_lock:
                    _tts_current["text"] = None
                    _tts_current["thread"] = None

        th = threading.Thread(target=_worker, args=(text,))
        th.daemon = True
        _tts_current["thread"] = th
        _tts_current["text"] = text
        th.start()


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI()


# Camera reconnection helper: try to reopen the capture device a few times
async def _try_reopen_camera(cap: Optional[cv2.VideoCapture], device: int = 0, attempts: int = 3, delay: float = 0.6):
    """Attempt to reopen the camera device and return a new VideoCapture or None.

    This is async-friendly and performs blocking ops with asyncio.to_thread.
    """
    try:
        if cap is not None:
            try:
                await asyncio.to_thread(cap.release)
            except Exception:
                logging.debug("Failed to release previous camera during reopen attempt", exc_info=True)
    except Exception:
        logging.debug("Error while releasing camera before reopen", exc_info=True)

    for attempt in range(1, attempts + 1):
        try:
            # Try DirectShow backend first (Windows), fallback to default
            try:
                new_cap = await asyncio.to_thread(cv2.VideoCapture, device, cv2.CAP_DSHOW)
            except Exception:
                new_cap = await asyncio.to_thread(cv2.VideoCapture, device)
            opened = await asyncio.to_thread(new_cap.isOpened)
            if opened:
                # warm up: wait for camera to stabilize and try reading a frame
                await asyncio.sleep(0.5)
                try:
                    ret, _ = await asyncio.to_thread(new_cap.read)
                    if ret:
                        logging.info("Reopened camera on attempt %d/%d (with frame validation)", attempt, attempts)
                        return new_cap
                    else:
                        logging.debug("Camera reopened but read failed (attempt %d), retrying", attempt)
                except Exception:
                    logging.debug("Exception during frame validation after reopen (attempt %d)", attempt, exc_info=True)
                # if we get here, frame validation failed, try to release and retry
                try:
                    await asyncio.to_thread(new_cap.release)
                except Exception:
                    pass
            else:
                try:
                    await asyncio.to_thread(new_cap.release)
                except Exception:
                    pass
        except Exception:
            logging.debug("Exception while trying to reopen camera (attempt %d)", attempt, exc_info=True)

        # wait before next attempt
        await asyncio.sleep(delay)

    logging.warning("Failed to reopen camera after %d attempts", attempts)
    return None


# -----------------------------
# SignModel wrapper
# -----------------------------
class SignModel:
    """Lightweight wrapper for a sign recognition model.

    Attempts to use a PyTorch LSTM model if available and a model file is
    present. Otherwise falls back to a small mock predictor that uses simple
    temporal heuristics on the landmarks sequence. The predict() method
    returns (label_str, confidence_float).
    """

    def __init__(self, model_path: str = MODEL_PATH, labels: dict = None):
        self.model_path = model_path
        self.use_torch = False
        self.torch = None
        self.model = None
        # default labels mapping
        self.labels = labels or {
            0: "Bonjour",
            1: "Merci",
            2: "Oui",
            3: "Non",
            4: "Au revoir",
        }

        # Try to import torch and load a small LSTM model if file present
        try:
            import torch
            import torch.nn as nn

            self.torch = torch

            class LSTMModel(nn.Module):
                def __init__(self, input_size=63, hidden_size=64, num_layers=1, num_classes=len(self.labels)):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_size, num_classes)

                def forward(self, x):
                    # x: (batch, seq_len, input_size)
                    out, _ = self.lstm(x)
                    # take last timestep
                    last = out[:, -1, :]
                    return self.fc(last)

            # If a model file exists, instantiate and load
            if os.path.exists(self.model_path):
                model = LSTMModel()
                state = torch.load(self.model_path, map_location="cpu")
                model.load_state_dict(state)
                model.eval()
                self.model = model
                self.use_torch = True
                logging.info("Loaded PyTorch sign model from %s", self.model_path)
            else:
                logging.info("No PyTorch model file found at %s — using mock predictor", self.model_path)
                self.use_torch = False

        except Exception:
            # Torch not available or load failed — fall back to mock
            logging.info("PyTorch not available or model load failed — using mock predictor")
            self.use_torch = False

    def predict(self, seq: List[List[Tuple[float, float, float]]]):
        """Predict a label from a sequence of frames.

        seq: list of frames, each frame is list of 21 (x,y,z) tuples.
        returns (label_str, confidence)
        """
        try:
            if self.use_torch and self.model is not None and self.torch is not None:
                # Prepare tensor (1, seq_len, 63)
                import numpy as _np

                arr = _np.array(seq, dtype=_np.float32).reshape(1, len(seq), 63)
                tensor = self.torch.from_numpy(arr)
                with self.torch.no_grad():
                    logits = self.model(tensor)
                    probs = self.torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
                    idx = int(probs.argmax())
                    return self.labels.get(idx, ""), float(probs[idx])

            # Mock predictor: use simple temporal heuristics across the sequence
            return self._mock_predict(seq)
        except Exception:
            logging.exception("Error during model prediction; falling back to mock")
            return self._mock_predict(seq)

    def _mock_predict(self, seq: List[List[Tuple[float, float, float]]]):
        """A simple but credible mock: uses thumb-index distance and motion.

        Returns a label and a confidence in [0.0, 1.0].
        """
        try:
            if not seq:
                return "", 0.0

            # compute mean thumb-index distance over sequence and motion
            thumb_index_dists = []
            centroids = []
            for frame in seq:
                xs = [p[0] for p in frame]
                ys = [p[1] for p in frame]
                centroids.append((sum(xs) / len(xs), sum(ys) / len(ys)))
                tx, ty, _ = frame[4]
                ix, iy, _ = frame[8]
                thumb_index_dists.append(((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5)

            mean_dist = sum(thumb_index_dists) / len(thumb_index_dists)
            # motion vector from first to last
            dx = centroids[-1][0] - centroids[0][0]
            dy = centroids[-1][1] - centroids[0][1]
            abs_dx = abs(dx)
            abs_dy = abs(dy)

            # crude thresholds (normalized coords)
            if abs_dy > abs_dx * 1.5 and abs_dy > 0.02:
                return "Oui", 0.85
            if abs_dx > abs_dy * 1.5 and abs_dx > 0.02:
                return "Non", 0.85
            # open vs closed via mean_dist
            if mean_dist > 0.08:
                return "Bonjour", 0.8
            if mean_dist < 0.035:
                return "Merci", 0.78
            # fallback
            return "", 0.0
        except Exception:
            logging.exception("Mock predictor failed")
            return "", 0.0


# instantiate model at module import so it's ready for connections
sign_model = SignModel()

# --- Simple signs database API (SQLite) ------------------------------
from pydantic import BaseModel
from typing import Any
from . import signs_db


class SignCreate(BaseModel):
    label: str
    aliases: Optional[list] = None
    landmarks: Optional[Any] = None
    metadata: Optional[dict] = None


class SignUpdate(BaseModel):
    label: Optional[str] = None
    aliases: Optional[list] = None
    landmarks: Optional[Any] = None
    metadata: Optional[dict] = None


@app.get('/api/signs')
def api_list_signs():
    return {'signs': signs_db.get_all_signs()}


@app.get('/api/signs/{sign_id}')
def api_get_sign(sign_id: int):
    s = signs_db.get_sign(sign_id)
    if not s:
        return {'error': 'not_found'}
    return {'sign': s}


@app.post('/api/signs')
def api_create_sign(payload: SignCreate):
    sid = signs_db.create_sign(label=payload.label, aliases=payload.aliases, landmarks=payload.landmarks, metadata=payload.metadata)
    return {'id': sid}


@app.put('/api/signs/{sign_id}')
def api_update_sign(sign_id: int, payload: SignUpdate):
    ok = signs_db.update_sign(sign_id, label=payload.label, aliases=payload.aliases, landmarks=payload.landmarks, metadata=payload.metadata)
    return {'ok': bool(ok)}


@app.delete('/api/signs/{sign_id}')
def api_delete_sign(sign_id: int):
    ok = signs_db.delete_sign(sign_id)
    return {'ok': bool(ok)}


@app.get('/api/signs/export')
def api_export_signs():
    return {'signs': signs_db.export_all()}


@app.post('/api/signs/import')
def api_import_signs(payload: dict):
    signs = payload.get('signs') or []
    replace = bool(payload.get('replace'))
    inserted = signs_db.import_list(signs, replace=replace)
    return {'inserted': inserted}

# --------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_camera(ws: WebSocket):
    """Handle a websocket connection that streams camera summaries.

    The camera is opened on connect and closed on disconnect. MediaPipe Hands
    is initialized per-connection. For each frame we run MediaPipe and if at
    least one hand is detected we send a small JSON summary:

        {"hands_detected": <1|2>, "landmarks_count": 21, "timestamp": <ISO8601>}

    Rate limiting: we ensure at most 5 messages per second are sent
    (i.e., minimum 0.2s between sends).

    TODO: replace summary with an AI recognizer to return sign labels.
    """

    await ws.accept()

    logging.info("WebSocket client connected — opening camera and MediaPipe")

    cap: Optional[cv2.VideoCapture] = None
    mp_hands: Optional[object] = None

    try:
        # Open camera on connection (not at server startup)
        # On Windows, try DirectShow backend first for better stability
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        except Exception:
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logging.error("Could not open camera 0 for websocket client")
            await ws.send_json({"error": "could_not_open_camera"})
            await ws.close(code=1011)
            return

        logging.info("Camera opened successfully for websocket client")

        # Initialize MediaPipe Hands for this connection with requested params
        # Use the robustly imported MPHands class if available. If it's not
        # available on this system (e.g., mediapipe package not installed or
        # incompatible), fall back to a lightweight heartbeat-only mode and
        # keep the websocket open so clients can still receive camera status.
        if MPHands is None:
            # Attempt to initialize the Tasks API hand landmarker (if available)
            task_landmarker = _ensure_tasks_hand_landmarker()
            if task_landmarker is not None and TASK_IMAGE_MODULE is not None:
                logging.info("Using MediaPipe Tasks HandLandmarker for live detection")

                last_send_ts = 0.0
                last_text_ts = 0.0
                last_sent_text = ""
                last_sent_text_ts = 0.0

                from collections import deque

                history = deque(maxlen=6)
                subtitles_history = deque(maxlen=10)
                seq_buffer = deque(maxlen=SEQ_LEN)
                text_buffer = deque(maxlen=BUFFER_MAXLEN)

                # detection loop using Tasks API
                try:
                    detect_fail_count = 0
                    reopen_fail_count = 0  # track consecutive reopen failures to avoid infinite loops
                    while True:
                        try:
                            ret, frame = await asyncio.to_thread(cap.read)
                        except asyncio.CancelledError:
                            logging.info("Camera read cancelled (Tasks API) — stopping Tasks loop")
                            break
                        except Exception:
                            logging.exception("Exception while reading from camera with Tasks API — stopping")
                            break

                        if not ret:
                            logging.warning("Camera read returned ret=False — attempting to reopen camera (Tasks loop)")
                            # try to reopen the camera a few times before giving up
                            new_cap = await _try_reopen_camera(cap, device=0, attempts=4, delay=0.5)
                            if new_cap is None:
                                reopen_fail_count += 1
                                if reopen_fail_count >= 2:
                                    logging.error("Camera reopening failed %d times — stopping Tasks loop", reopen_fail_count)
                                    try:
                                        await ws.send_json({"error": "camera_lost"})
                                    except Exception:
                                        pass
                                    break
                                else:
                                    logging.warning("Reopen attempt failed, will retry on next read (fail_count=%d)", reopen_fail_count)
                                    await asyncio.sleep(1.0)  # back off before next read attempt
                                    continue
                            else:
                                cap = new_cap
                                reopen_fail_count = 0  # reset on successful reopen
                                # continue to next iteration to read from reopened camera
                                continue

                        # convert BGR->RGB and to uint8
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        try:
                            img = TASK_IMAGE_MODULE.Image(TASK_IMAGE_MODULE.ImageFormat.SRGB, rgb)
                        except Exception:
                            logging.exception("Failed to create Tasks Image from frame — skipping frame")
                            await asyncio.sleep(0.01)
                            continue

                        # Run detection in a thread because it may call native code
                        try:
                            # use detect_for_video without region_of_interest because
                            # this task build may not support ROI. Passing opts with
                            # ROI causes `ValueError: This task doesn't support region-of-interest.`
                            import time

                            timestamp_ms = int(time.time() * 1000)
                            # Call detect_for_video without ImageProcessingOptions
                            res = await asyncio.to_thread(task_landmarker.detect_for_video, img, timestamp_ms)
                            # reset failure counter on success
                            detect_fail_count = 0
                        except Exception:
                            # Rate-limit noisy repeated exceptions to avoid log flooding
                            detect_fail_count += 1
                            if detect_fail_count <= 5 or (detect_fail_count % 200) == 0:
                                logging.exception("Tasks hand_landmarker.detect_for_video failed for this frame — skipping")
                            else:
                                logging.debug("Tasks detect_for_video failing (count=%d); suppressing repeated trace", detect_fail_count)
                            await asyncio.sleep(0.01)
                            continue

                        # Convert result to list of landmarks: choose first hand if any
                        landmarks = []
                        try:
                            if getattr(res, 'hand_landmarks', None):
                                # res.hand_landmarks is a list of lists of NormalizedLandmark
                                hand0 = res.hand_landmarks[0]
                                for lm in hand0:
                                    # each lm has x,y,z attributes
                                    landmarks.append((float(lm.x), float(lm.y), float(getattr(lm, 'z', 0.0))))
                        except Exception:
                            logging.exception("Failed to parse Tasks HandLandmarker result")

                        now = asyncio.get_event_loop().time()

                        # If landmarks found, push to seq buffer and run recognition
                        if landmarks:
                            seq_buffer.append(landmarks)
                            # only predict when we have some frames
                            if len(seq_buffer) >= 3:
                                try:
                                    label, conf = sign_model.predict(list(seq_buffer))
                                except Exception:
                                    logging.exception("sign_model.predict failed")
                                    label, conf = "", 0.0

                                if label and conf >= CONFIDENCE_THRESHOLD:
                                    # record candidate
                                    text_buffer.append((label, now))
                                    # Inline stability check because is_candidate_stable
                                    # is defined later in the file (out of this scope).
                                    cutoff = now - STABILITY_WINDOW
                                    count = 0
                                    for ttxt, tts in reversed(text_buffer):
                                        if tts < cutoff:
                                            break
                                        if ttxt == label:
                                            count += 1
                                    stable = (count >= STABILITY_N)
                                    frames_used = count
                                    if stable:
                                        # enforce repetition cooldown
                                        if label != last_sent_text or (now - last_sent_text_ts) > REPETITION_COOLDOWN:
                                            payload = {
                                                "text": label,
                                                "confidence": conf,
                                                "streaming": {"stable": True, "frames_used": frames_used},
                                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                            }
                                            try:
                                                await ws.send_json(payload)
                                            except WebSocketDisconnect:
                                                logging.info("WebSocket client disconnected during Tasks send")
                                                break
                                            except Exception:
                                                logging.exception("Failed to send detection payload to client")
                                            last_sent_text = label
                                            last_sent_text_ts = now
                                            subtitles_history.append(label)
                                            if TTS_ENABLED:
                                                speak_async(label)

                        # Periodic lightweight heartbeat/messages (rate-limited)
                        if now - last_send_ts >= 0.2:
                            iso_ts = datetime.now(timezone.utc).isoformat()
                            try:
                                await ws.send_json({"hands_detected": 1 if landmarks else 0, "timestamp": iso_ts})
                            except WebSocketDisconnect:
                                logging.info("WebSocket client disconnected during Tasks heartbeat")
                                break
                            except Exception:
                                logging.exception("Failed to send Tasks heartbeat to client")
                                break
                            last_send_ts = now

                        await asyncio.sleep(0.01)
                finally:
                    # ensure any resources are cleaned up by outer finally
                    pass
                return
            else:
                logging.warning("MediaPipe Hands is not available; running in heartbeat-only mode")

                # Heartbeat loop: send a simple "camera active" JSON every 1s
                last_sent = 0.0
                try:
                    while True:
                        try:
                            ret, _ = await asyncio.to_thread(cap.read)
                        except asyncio.CancelledError:
                            logging.info("Camera read cancelled (heartbeat mode) — stopping heartbeat loop")
                            break
                        except Exception:
                            logging.exception("Exception while reading from camera in heartbeat mode — stopping")
                            break

                        if not ret:
                            logging.warning("Camera read returned ret=False in heartbeat mode — attempting to reopen")
                            new_cap = await _try_reopen_camera(cap, device=0, attempts=3, delay=0.5)
                            if new_cap is None:
                                logging.error("Could not reopen camera in heartbeat mode — stopping")
                                break
                            cap = new_cap
                            continue

                        now = asyncio.get_event_loop().time()
                        if now - last_sent >= 1.0:
                            iso_ts = datetime.now(timezone.utc).isoformat()
                            try:
                                await ws.send_json({"text": "camera active", "timestamp": iso_ts})
                            except WebSocketDisconnect:
                                logging.info("WebSocket client disconnected in heartbeat mode")
                                break
                            except Exception:
                                logging.exception("Failed to send heartbeat to client in heartbeat mode")
                                break
                            last_sent = now

                        await asyncio.sleep(0.01)
                finally:
                    # Release camera in the outer finally block below
                    pass
                return

        mp_hands = MPHands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        # Timestamp of last sent summary message (for rate limiting summaries)
        last_send_ts = 0.0
        # Timestamp of last sent text message (for rate limiting text outputs)
        last_text_ts = 0.0
        # Last text sent and its timestamp to enforce anti-spam for same text
        last_sent_text = ""
        last_sent_text_ts = 0.0

        # History buffers (used to compute stability and motion). We keep a small
        # sliding window of recent mean positions and thumb-index distances.
        from collections import deque

        history = deque(maxlen=6)  # store recent frames summaries (small buffer)
        # Subtitles history (last 10 recognized texts) — per-connection
        subtitles_history = deque(maxlen=10)
        # Sequence buffer for model input: store recent frames' landmarks
        seq_buffer = deque(maxlen=SEQ_LEN)
        # Per-frame recognized text buffer (text, timestamp) used for stability
        text_buffer = deque(maxlen=BUFFER_MAXLEN)

        # Helper: check if a candidate text is stable in the recent buffer
        def is_candidate_stable(candidate: str, now_ts: float):
            """Return (is_stable, frames_used).

            Count occurrences of candidate in text_buffer within STABILITY_WINDOW
            seconds from now_ts. If count >= STABILITY_N, return True and the
            number of matching frames used; otherwise return False.
            """
            if not candidate:
                return False, 0

            cutoff = now_ts - STABILITY_WINDOW
            count = 0
            # frames_used = number of entries matching candidate in the window
            for t, ts in reversed(text_buffer):
                # entries are (text, timestamp)
                if ts < cutoff:
                    break
                if t == candidate:
                    count += 1
            return (count >= STABILITY_N, count)

        # Define the simulated recognizer function as a closure so it can use
        # the per-connection `history` for stability checks. Signature matches
        # the requirement: recognize_sign(landmarks: list) -> str
        def recognize_sign(landmarks: list) -> str:
            """Simulated sign recognizer based on simple geometric heuristics.

            Input:
                landmarks: list of 21 landmarks, each with (x, y, z) normalized

            Returns:
                A short text label (e.g., 'Bonjour', 'Merci', 'Oui', 'Non') or
                an empty string if no pattern matched.

            Heuristics implemented:
                - mean_x, mean_y: centroid of landmarks
                - hand_size: approx size of the hand (bbox diagonal)
                - thumb_index_dist: euclidean distance between landmarks 4 and 8
                - stability: low variance of centroid over recent frames
                - motion: dominant direction of centroid change over recent frames

            TODO: Remplacer recognize_sign par un modèle ML plus tard.
            """

            try:
                if not landmarks or len(landmarks) < 21:
                    return ""

                # compute centroid (mean x,y)
                xs = [p[0] for p in landmarks]
                ys = [p[1] for p in landmarks]
                mean_x = sum(xs) / len(xs)
                mean_y = sum(ys) / len(ys)

                # compute simple hand size as diagonal of bounding box
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                dx = max_x - min_x
                dy = max_y - min_y
                hand_size = (dx * dx + dy * dy) ** 0.5

                # thumb (4) and index (8) tips
                tx, ty, _ = landmarks[4]
                ix, iy, _ = landmarks[8]
                thumb_index_dist = ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5

                # update history with this frame's summary
                history.append({
                    "mean_x": mean_x,
                    "mean_y": mean_y,
                    "thumb_index_dist": thumb_index_dist,
                    "hand_size": hand_size,
                })

                # compute stability as std dev of mean positions in history
                if len(history) >= 3:
                    mean_xs = [h["mean_x"] for h in history]
                    mean_ys = [h["mean_y"] for h in history]
                    avg_x = sum(mean_xs) / len(mean_xs)
                    avg_y = sum(mean_ys) / len(mean_ys)
                    var_x = sum((v - avg_x) ** 2 for v in mean_xs) / len(mean_xs)
                    var_y = sum((v - avg_y) ** 2 for v in mean_ys) / len(mean_ys)
                    stability = (var_x + var_y) ** 0.5
                else:
                    stability = float("inf")

                # compute average motion vector over history
                motion_dx = 0.0
                motion_dy = 0.0
                if len(history) >= 2:
                    # use first and last to get overall motion
                    motion_dx = history[-1]["mean_x"] - history[0]["mean_x"]
                    motion_dy = history[-1]["mean_y"] - history[0]["mean_y"]

                # Normalize thresholds relative to hand_size (which is in normalized coords)
                size = max(hand_size, 1e-6)
                open_threshold = 0.4 * size
                closed_threshold = 0.18 * size
                motion_threshold = 0.02  # small absolute threshold for motion

                # Heuristics for signs
                is_open = thumb_index_dist >= open_threshold
                is_closed = thumb_index_dist <= closed_threshold
                abs_dx = abs(motion_dx)
                abs_dy = abs(motion_dy)

                # Movement dominant in vertical direction
                if abs_dy > abs_dx * 1.5 and abs_dy > motion_threshold:
                    return "Oui"

                # Movement dominant in horizontal direction
                if abs_dx > abs_dy * 1.5 and abs_dx > motion_threshold:
                    return "Non"

                # Open and stable -> Bonjour
                if is_open and stability < 1e-4:
                    return "Bonjour"

                # Closed -> Merci
                if is_closed:
                    return "Merci"

                # No pattern matched
                return ""

            except Exception:
                # Any error in heuristics should not break the server
                logging.exception("Error in recognize_sign heuristics")
                return ""

        # Main loop: read frames, process with MediaPipe, send summaries
        while True:
            # Read frame off the event loop to avoid blocking
            try:
                ret, frame = await asyncio.to_thread(cap.read)
            except asyncio.CancelledError:
                logging.info("Camera read cancelled — stopping main loop")
                break
            except Exception:
                logging.exception("Exception while reading from camera — stopping loop")
                break

            if not ret:
                logging.warning("Camera read returned ret=False — attempting to reopen main camera")
                new_cap = await _try_reopen_camera(cap, device=0, attempts=3, delay=0.5)
                if new_cap is None:
                    logging.error("Could not reopen camera — stopping main loop")
                    break
                cap = new_cap
                continue

            # Process frame with MediaPipe (run in thread because it's CPU-bound)
            try:
                frame_rgb = await asyncio.to_thread(cv2.cvtColor, frame, cv2.COLOR_BGR2RGB)
                results = await asyncio.to_thread(mp_hands.process, frame_rgb)
            except Exception:
                # If MediaPipe fails on this frame, ignore frame and continue
                logging.exception("MediaPipe processing failed for this frame — skipping")
                await asyncio.sleep(0)  # yield
                continue

            # If hands detected, prepare a small summary message and run recognition
            if results and getattr(results, "multi_hand_landmarks", None):
                try:
                    hands_detected = len(results.multi_hand_landmarks)
                except Exception:
                    hands_detected = 0

                # For each detected hand, extract landmarks and try to recognize a sign.
                # We'll only run recognition on the first hand to keep computations light.
                try:
                    if hands_detected > 0:
                        # extract landmarks for the first hand: list of (x,y,z)
                        first_hand = results.multi_hand_landmarks[0]
                        landmarks = [(p.x, p.y, p.z) for p in first_hand.landmark]

                        # Append landmarks to sequence buffer for model inference
                        seq_buffer.append(landmarks)

                        # Run the sign model on the current sequence buffer. The
                        # model accepts variable-length sequences; we only act on
                        # predictions that cross the confidence threshold.
                        label, conf = "", 0.0
                        try:
                            label, conf = sign_model.predict(list(seq_buffer))
                        except Exception:
                            logging.exception("Sign model prediction failed; skipping this frame")

                        now = asyncio.get_event_loop().time()

                        # If model predicts a label with sufficient confidence,
                        # append to text buffer for temporal stability checks
                        if label and conf >= CONFIDENCE_THRESHOLD:
                            text_buffer.append((label, now))

                            # Validate candidate stability using the same buffer logic
                            try:
                                is_stable, frames_used = is_candidate_stable(label, now)
                            except Exception:
                                logging.exception("Error while validating candidate stability")
                                is_stable, frames_used = False, 0

                            if is_stable:
                                # Enforce cooldown for identical emitted texts
                                if label == last_sent_text and (now - last_sent_text_ts) < REPETITION_COOLDOWN:
                                    pass
                                else:
                                    # Enforce global text rate limit: max ~3 msgs/sec
                                    if (now - last_text_ts) < (1.0 / 3.0):
                                        pass
                                    else:
                                        # Emit final validated text + streaming info and include model confidence
                                        iso_ts = datetime.now(timezone.utc).isoformat()
                                        if not subtitles_history or subtitles_history[-1] != label:
                                            subtitles_history.append(label)

                                        text_payload = {
                                            "text": label,
                                            "confidence": float(conf),
                                            "history": list(subtitles_history),
                                            "streaming": {"stable": True, "frames_used": frames_used},
                                            "timestamp": iso_ts,
                                        }
                                        try:
                                            await ws.send_json(text_payload)
                                            logging.info("Sent validated text to client: %s", text_payload)
                                            last_text_ts = now
                                            last_sent_text = label
                                            last_sent_text_ts = now
                                            if TTS_ENABLED:
                                                speak_async(label)
                                        except WebSocketDisconnect:
                                            logging.info("WebSocket client disconnected while sending validated text")
                                            break
                                        except Exception:
                                            logging.exception("Failed to send validated text to client")
                                            # continue

                    # Send the summarized hands message (rate-limited separately)
                    if hands_detected > 0:
                        landmarks_count = 21
                        now = asyncio.get_event_loop().time()
                        if now - last_send_ts >= 0.2:
                            iso_ts = datetime.now(timezone.utc).isoformat()
                            payload = {
                                "hands_detected": hands_detected,
                                "landmarks_count": landmarks_count,
                                "timestamp": iso_ts,
                            }
                            try:
                                await ws.send_json(payload)
                                logging.info("Sent hands summary to client: %s", payload)
                            except WebSocketDisconnect:
                                logging.info("WebSocket client disconnected while sending")
                                break
                            except Exception:
                                logging.exception("Failed to send hands summary to client — stopping loop")
                                break

                            last_send_ts = now
                except Exception:
                    logging.exception("Error during recognition/send pipeline — continuing")

            # Yield to event loop briefly to avoid busy loop; small sleep keeps loop responsive
            await asyncio.sleep(0)

    except WebSocketDisconnect:
        logging.info("WebSocket client disconnected")
    except Exception:
        logging.exception("Unexpected error in websocket handler")
    finally:
        # Cleanup: ensure MediaPipe and camera are closed/released
        if mp_hands is not None:
            try:
                mp_hands.close()
            except Exception:
                logging.exception("Error closing MediaPipe Hands")

        if cap is not None:
            try:
                cap.release()
                logging.info("Camera released")
            except Exception:
                logging.exception("Error releasing camera")
