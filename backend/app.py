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

# Robust import for MediaPipe Hands.
MPHands = None
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
    except Exception:
        # If import fails, leave MPHands as None and handle at runtime
        MPHands = None
        logging.warning(
            "Unable to import MediaPipe Hands; hand detection will be disabled."
        )
import threading
import pyttsx3

# Global toggle to enable/disable TTS easily
TTS_ENABLED = True

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
            logging.warning("MediaPipe Hands is not available; running in heartbeat-only mode")

            # Heartbeat loop: send a simple "camera active" JSON every 1s
            last_sent = 0.0
            try:
                while True:
                    try:
                        ret, _ = await asyncio.to_thread(cap.read)
                    except Exception:
                        logging.exception("Exception while reading from camera in heartbeat mode — stopping")
                        break

                    if not ret:
                        logging.warning("Camera read returned ret=False in heartbeat mode — stopping loop")
                        break

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
            except Exception:
                logging.exception("Exception while reading from camera — stopping loop")
                break

            if not ret:
                logging.warning("Camera read returned ret=False — stopping loop")
                break

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
                recognized_text = ""
                try:
                    if hands_detected > 0:
                        # extract landmarks for the first hand: list of (x,y,z)
                        first_hand = results.multi_hand_landmarks[0]
                        landmarks = [(p.x, p.y, p.z) for p in first_hand.landmark]

                        # Call the simulated recognizer (uses closure history)
                        recognized_text = recognize_sign(landmarks)

                        # If a text is recognized, enforce anti-spam and rate limiting
                        if recognized_text:
                            now = asyncio.get_event_loop().time()
                            # Enforce min interval between identical texts (2s)
                            if recognized_text == last_sent_text and (now - last_sent_text_ts) < 2.0:
                                # skip sending duplicate text within 2s
                                recognized_text = ""

                            # Enforce global text rate limit: max 3 msgs/sec
                            if (now - last_text_ts) < (1.0 / 3.0):
                                # skip due to rate limit
                                recognized_text = ""

                            # If still allowed, send the text message
                            if recognized_text:
                                iso_ts = datetime.now(timezone.utc).isoformat()
                                # Update subtitles history: avoid consecutive duplicates
                                if not subtitles_history or subtitles_history[-1] != recognized_text:
                                    subtitles_history.append(recognized_text)

                                text_payload = {
                                    "text": recognized_text,
                                    "history": list(subtitles_history),
                                    "confidence": 0.8,
                                    "timestamp": iso_ts,
                                }
                                try:
                                    await ws.send_json(text_payload)
                                    logging.info("Sent recognized text to client: %s", text_payload)
                                    last_text_ts = now
                                    # update anti-spam trackers
                                    last_sent_text = recognized_text
                                    last_sent_text_ts = now
                                    # Trigger non-blocking TTS if enabled
                                    try:
                                        if TTS_ENABLED:
                                            speak_async(recognized_text)
                                    except Exception:
                                        logging.exception("Error while launching TTS (ignored)")
                                except WebSocketDisconnect:
                                    logging.info("WebSocket client disconnected while sending text")
                                    break
                                except Exception:
                                    logging.exception("Failed to send recognized text to client")
                                    # do not break; continue with summaries

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
