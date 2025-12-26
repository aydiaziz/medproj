"""FastAPI WebSocket server with centralized camera management.

ARCHITECTURE:
- 1 caméra globale = 1 ouverture = 1 thread dédié
- Le thread caméra lit les frames en continu et les stocke
- Le WebSocket lit la dernière frame disponible (pas de gestion caméra)
- Nettoyage propre à l'arrêt du serveur uniquement

Usage:
    uvicorn app_new:app --host 127.0.0.1 --port 8000

"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional
from collections import deque
import numpy as np
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import cv2

# ============================================================================
# MEDIAPIPE IMPORTS avec fallback robuste
# ============================================================================

TASK_HAND_LANDMARKER = None
TASK_IMAGE_MODULE = None
_TASK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
MEDIAPIPE_AVAILABLE = False

try:
    # Essayer d'importer via le chemin standard Tasks API
    from mediapipe.tasks.python.vision import hand_landmarker as hl
    from mediapipe.tasks.python.vision.core import image as image_lib
    from mediapipe.tasks.python.vision.core import vision_task_running_mode
    TASK_IMAGE_MODULE = image_lib
    MEDIAPIPE_AVAILABLE = True
    logging.info("MediaPipe Tasks API importé avec succès")
except ImportError as e:
    logging.warning(f"MediaPipe Tasks API non disponible: {e}")
    MEDIAPIPE_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================
CAMERA_DEVICE = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
FRAME_READ_TIMEOUT = 0.5  # timeout pour lire une frame

# Stabilité
STABILITY_N = 5
STABILITY_WINDOW = 2.0
REPETITION_COOLDOWN = 3.0
BUFFER_MAXLEN = 15

# MediaPipe
HAND_LANDMARKER_MODEL = "hand_landmarker.task"
MIN_HAND_CONFIDENCE = 0.5  # Minimum confidence pour détecter une main

# ============================================================================
# CONFIGURATION LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ============================================================================
# MEDIAPIPE HANDS DETECTOR
# ============================================================================

class HandLandmarkerWrapper:
    """Wrapper pour MediaPipe HandLandmarker Tasks API avec gestion d'erreurs."""
    
    def __init__(self, model_path: str = _TASK_MODEL_PATH):
        self.model_path = model_path
        self.landmarker = None
        self.lock = threading.RLock()
        self.init_error: Optional[str] = None
        self._init_landmarker()
    
    def _init_landmarker(self):
        """Initialiser le HandLandmarker MediaPipe Tasks API."""
        if not MEDIAPIPE_AVAILABLE or TASK_IMAGE_MODULE is None:
            self.init_error = "MediaPipe Tasks API non disponible"
            logging.warning(self.init_error)
            return
        
        try:
            # Vérifier que le modèle existe
            if not os.path.exists(self.model_path):
                self.init_error = f"Modèle {self.model_path} non trouvé"
                logging.warning(self.init_error)
                return
            
            # Importer HandLandmarker
            from mediapipe.tasks.python.vision import hand_landmarker as hl
            
            # Créer HandLandmarkerOptions avec VIDEO mode
            try:
                options = hl.HandLandmarkerOptions(
                    base_options=hl._BaseOptions(model_asset_path=self.model_path),
                    running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
                    min_hand_detection_confidence=MIN_HAND_CONFIDENCE,
                    num_hands=2
                )
                self.landmarker = hl.HandLandmarker.create_from_options(options)
                logging.info(f"HandLandmarker initialisé (VIDEO mode) avec {self.model_path}")
            except Exception as e:
                # Fallback: IMAGE mode
                logging.warning(f"VIDEO mode échoué, fallback à IMAGE mode: {e}")
                self.landmarker = hl.HandLandmarker.create_from_model_path(self.model_path)
                logging.info(f"HandLandmarker initialisé (IMAGE mode) avec {self.model_path}")
        
        except Exception as e:
            self.init_error = f"Erreur lors de l'initialisation HandLandmarker: {e}"
            logging.exception(self.init_error)
            self.landmarker = None
    
    def detect(self, frame: np.ndarray, frame_count: int) -> dict:
        """Détecter les landmarks des mains dans une frame.
        
        Args:
            frame: Image BGR (OpenCV format)
            frame_count: Numéro de frame (timestamp pour Tasks API VIDEO mode)
        
        Returns:
            dict avec landmarks, scores, etc. ou {} si pas de mains détectées
        """
        if self.landmarker is None:
            return {}
        
        try:
            with self.lock:
                # Convertir BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Créer mp_image
                mp_image = TASK_IMAGE_MODULE.Image(
                    image_format=TASK_IMAGE_MODULE.ImageFormat.SRGB,
                    data=frame_rgb
                )
                
                # Détecter (VIDEO mode)
                result = self.landmarker.detect_for_video(mp_image, frame_count)
            
            # Extraire les données
            output = {
                "num_hands": 0,
                "hands": []
            }
            
            # result.handedness et result.hand_landmarks sont des listes
            if result.handedness and result.hand_landmarks:
                num_hands = len(result.handedness)
                output["num_hands"] = num_hands
                
                for i in range(num_hands):
                    # Accéder aux éléments via index
                    handedness_list = result.handedness[i]  # C'est une liste de Categories
                    hand_landmarks_list = result.hand_landmarks[i]  # C'est une liste de Landmarks
                    
                    # Récupérer le premier élément (c'est une liste)
                    if handedness_list and len(handedness_list) > 0:
                        handedness_obj = handedness_list[0]
                        label = getattr(handedness_obj, 'category_name', f'Hand{i}')
                        score = float(getattr(handedness_obj, 'score', 1.0))
                    else:
                        label = f'Hand{i}'
                        score = 1.0
                    
                    hand_data = {
                        "index": i,
                        "label": label,
                        "confidence": score,
                        "landmarks": []
                    }
                    
                    # Extraire les landmarks
                    if hand_landmarks_list:
                        for landmark in hand_landmarks_list:
                            presence = getattr(landmark, "presence", None)
                            presence = float(presence) if presence is not None else 1.0
                            hand_data["landmarks"].append({
                                "x": float(landmark.x),
                                "y": float(landmark.y),
                                "z": float(landmark.z),
                                "presence": presence
                            })
                    
                    output["hands"].append(hand_data)
            
            return output
        
        except Exception as e:
            logging.exception(f"Erreur détection mains: {e}")
            return {}
    
    def is_ready(self) -> bool:
        """Vérifier si le landmarker est prêt."""
        return self.landmarker is not None


# Instance globale HandLandmarker
hand_detector = HandLandmarkerWrapper()

# ============================================================================
# CAMÉRA GLOBALE — Architecture centralisée
# ============================================================================

class CameraManager:
    """Gestionnaire caméra global avec thread dédié et lock.
    
    Garantit:
    - Une seule ouverture caméra
    - Une seule fermeture (à l'arrêt du serveur)
    - Lecture continue en background thread
    - Accès thread-safe à la dernière frame valide
    """
    
    def __init__(self, device: int = 0, width: int = 640, height: int = 480):
        self.device = device
        self.width = width
        self.height = height
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.lock = threading.RLock()
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_time: float = 0.0
        self.last_detection: dict = {}  # Résultats MediaPipe
        self.frame_count: int = 0
        
        self.running = False
        self.reader_thread: Optional[threading.Thread] = None
        self.init_error: Optional[str] = None
    
    def init(self) -> bool:
        """Initialiser la caméra UNE SEULE FOIS au démarrage du serveur."""
        with self.lock:
            if self.cap is not None:
                logging.info("Caméra déjà initialisée")
                return True
            
            try:
                # Essayer DirectShow (Windows) pour meilleure stabilité
                try:
                    self.cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW)
                except Exception:
                    self.cap = cv2.VideoCapture(self.device)
                
                if not self.cap.isOpened():
                    self.init_error = "Impossible d'ouvrir la caméra"
                    logging.error(self.init_error)
                    self.cap = None
                    return False
                
                # Configurer résolution et FPS
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                
                # Optionnel: buffers peu profonds (réduire latence)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                logging.info(
                    "Caméra initialisée: device=%d, %dx%d, %d FPS",
                    self.device, self.width, self.height, CAMERA_FPS
                )
                
                # Démarrer le thread de lecture
                self.running = True
                self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
                self.reader_thread.start()
                logging.info("Thread caméra démarré")
                
                return True
            
            except Exception as e:
                self.init_error = f"Erreur lors de l'initialisation caméra: {e}"
                logging.exception(self.init_error)
                self.cap = None
                return False
    
    def _reader_loop(self):
        """Thread dédié: lire les frames en continu et stocker la dernière.
        
        Ce thread tourne indépendamment des connexions WebSocket.
        Il lit les frames aussi vite que possible, les stocke, et détecte les mains.
        """
        logging.info("Reader loop démarré")
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            try:
                with self.lock:
                    if self.cap is None or not self.cap.isOpened():
                        logging.warning("Caméra fermée, arrêt du reader loop")
                        break
                    
                    ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # Frame valide: la stocker
                    with self.lock:
                        self.last_frame = frame.copy()  # copie pour éviter race conditions
                        self.last_frame_time = time.time()
                        self.frame_count += 1
                        frame_count = self.frame_count
                    
                    # === Détection MediaPipe (non-blocking) ===
                    if hand_detector.is_ready():
                        try:
                            detection_result = hand_detector.detect(frame, frame_count)
                            with self.lock:
                                self.last_detection = detection_result
                        except Exception as e:
                            logging.warning(f"Erreur détection MediaPipe: {e}")
                    
                    consecutive_failures = 0
                    
                    # Throttle pour ne pas lire plus vite que nécessaire
                    time.sleep(1.0 / CAMERA_FPS)
                
                else:
                    # ret=False: frame non valide
                    consecutive_failures += 1
                    logging.warning(
                        "Lecture caméra échouée (tentative %d/%d)",
                        consecutive_failures, max_failures
                    )
                    
                    if consecutive_failures >= max_failures:
                        logging.error("Trop d'échecs de lecture, arrêt du reader loop")
                        break
                    
                    # Petit délai avant prochaine tentative
                    time.sleep(0.1)
            
            except Exception as e:
                logging.exception("Exception dans reader loop: %s", e)
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logging.error("Reader loop arrêté après %d erreurs", max_failures)
                    break
                time.sleep(0.1)
        
        logging.info("Reader loop terminé (frames lues: %d)", self.frame_count)
    
    def read_frame(self) -> Optional[tuple]:
        """Retourner la dernière frame valide (non-blocking).
        
        Returns:
            (frame, timestamp, detection) ou None si pas de frame disponible
        """
        with self.lock:
            if self.last_frame is None:
                return None
            return (self.last_frame.copy(), self.last_frame_time, self.last_detection.copy())
    
    def get_detection(self) -> dict:
        """Retourner la dernière détection MediaPipe."""
        with self.lock:
            return self.last_detection.copy()
    
    def is_ready(self) -> bool:
        """Vérifier si la caméra est prête et a produit au moins une frame."""
        with self.lock:
            return self.cap is not None and self.cap.isOpened() and self.last_frame is not None
    
    def get_frame_count(self) -> int:
        """Retourner le nombre total de frames lues."""
        with self.lock:
            return self.frame_count
    
    def shutdown(self):
        """Arrêter le thread et fermer la caméra (appelé à l'arrêt du serveur)."""
        logging.info("Arrêt du gestionnaire caméra...")
        
        self.running = False
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=2.0)
        
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                logging.info("Caméra fermée")


# Instance globale unique
camera_manager = CameraManager(
    device=CAMERA_DEVICE,
    width=CAMERA_WIDTH,
    height=CAMERA_HEIGHT
)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Sign Recognition Backend")


@app.on_event("startup")
async def on_startup():
    """Initialiser la caméra et MediaPipe au démarrage du serveur."""
    logging.info("=== Démarrage du serveur ===")
    
    # Initialiser MediaPipe HandLandmarker
    if not hand_detector.is_ready():
        logging.warning("HandLandmarker non initialisé: %s", hand_detector.init_error)
    else:
        logging.info("MediaPipe HandLandmarker prêt")
    
    # Initialiser la caméra
    if not camera_manager.init():
        logging.error("ERREUR: Impossible d'initialiser la caméra")
        logging.error("Message: %s", camera_manager.init_error)
    else:
        logging.info("Caméra initialisée avec succès")


@app.on_event("shutdown")
async def on_shutdown():
    """Fermer la caméra à l'arrêt du serveur."""
    logging.info("=== Arrêt du serveur ===")
    camera_manager.shutdown()


@app.get("/health")
async def health_check():
    """Vérifier l'état de l'application."""
    return {
        "status": "ok",
        "camera_ready": camera_manager.is_ready(),
        "frames_read": camera_manager.get_frame_count(),
    }


@app.websocket("/ws")
async def websocket_camera(ws: WebSocket):
    """WebSocket passif: lire les frames disponibles et envoyer les résultats.
    
    IMPORTANT:
    - NE PAS ouvrir/fermer la caméra ici
    - Lire uniquement la dernière frame disponible
    - Envoyer détections MediaPipe au client via WebSocket
    """
    await ws.accept()
    logging.info("WebSocket client connecté")
    
    try:
        # Variables locales pour cette connexion
        last_send_ts = 0.0
        last_detection_sent = {}
        
        # Envoyer un message d'accueil
        await ws.send_json({
            "status": "connected",
            "camera_ready": camera_manager.is_ready(),
            "mediapipe_ready": hand_detector.is_ready(),
            "message": "Connecté au serveur, en attente de frames"
        })
        
        # Boucle principale: lire les frames et détections
        frame_skip = 0
        while True:
            # Vérifier si la caméra est prête
            if not camera_manager.is_ready():
                # Caméra pas encore prête, attendre
                await asyncio.sleep(0.1)
                continue
            
            # Lire la dernière frame disponible (non-blocking)
            frame_data = camera_manager.read_frame()
            if frame_data is None:
                # Pas de frame disponible, attendre un peu
                await asyncio.sleep(0.05)
                continue
            
            frame, frame_time, detection = frame_data
            
            # Throttle: traiter une frame sur N pour réduire charge CPU
            frame_skip += 1
            if frame_skip < 2:  # traiter 1 frame sur 2 (~15 FPS)
                continue
            frame_skip = 0
            
            now = time.time()
            
            # Envoyer les résultats MediaPipe
            if now - last_send_ts >= 0.1:  # ~10 Hz pour WebSocket
                # Construire le message avec détection
                message = {
                    "type": "detection",
                    "frames_total": camera_manager.get_frame_count(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "detection": detection
                }
                
                # Envoyer au client
                try:
                    await ws.send_json(message)
                except Exception as e:
                    logging.warning(f"Erreur envoi WebSocket: {e}")
                    break
                
                last_send_ts = now
            
            # Courte pause pour ne pas bloquer
            await asyncio.sleep(0.01)
    
    except WebSocketDisconnect:
        logging.info("WebSocket client déconnecté normalement")
    except Exception as e:
        logging.exception(f"Erreur WebSocket: {e}")
    finally:
        logging.info("WebSocket fermé (caméra continue de tourner)")


@app.get("/")
async def serve_frontend():
    """Servir le frontend HTML."""
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    try:
        with open(frontend_path, "r", encoding="utf-8") as f:
            content = f.read()
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return {"error": "Frontend not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
