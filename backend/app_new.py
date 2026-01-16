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
from typing import Optional, List, Dict
from collections import deque
import numpy as np
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import cv2

# ============================================================================
# SIGNS DATABASE IMPORT
# ============================================================================
from . import signs_db

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

# Sign Recognition
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence pour reconnaître un signe
SEQ_LEN = 10  # Nombre de frames pour la séquence de reconnaissance

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
                
                logging.debug(f"[MEDIAPIPE] {num_hands} main(s) détectée(s) à frame {frame_count}")
                
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
            logging.exception(f"[MEDIAPIPE_ERROR] Erreur détection: {e}")
            return {}
    
    def is_ready(self) -> bool:
        """Vérifier si le landmarker est prêt."""
        return self.landmarker is not None


# Instance globale HandLandmarker
hand_detector = HandLandmarkerWrapper()

# ============================================================================
# SIGN RECOGNIZER — Base de données des signes
# ============================================================================

class SignRecognizer:
    """Reconnaissance de signes basée sur la base de données.
    
    Compare les landmarks détectés avec les signes stockés dans la base de données.
    Utilise une métrique de distance pour trouver le signe le plus proche.
    """
    
    def __init__(self):
        self.signs_cache = []
        self._load_signs()
    
    def _load_signs(self):
        """Charger tous les signes depuis la base de données."""
        try:
            self.signs_cache = signs_db.get_all_signs()
            logging.info(f"[SIGNS] Chargé {len(self.signs_cache)} signes")
            for sign in self.signs_cache:
                logging.debug(f"[SIGNS] - {sign.get('label', 'N/A')} avec {len(sign.get('landmarks_json', []))} landmarks")
        except Exception as e:
            logging.exception(f"[SIGNS_ERROR] Erreur chargement: {e}")
            self.signs_cache = []
    
    def _calculate_distance(self, landmarks1: List[Dict], landmarks2: List[Dict]) -> float:
        """Calculer la distance euclidienne entre deux ensembles de landmarks.
        
        Args:
            landmarks1: Liste de dicts avec 'x', 'y', 'z'
            landmarks2: Liste de dicts avec 'x', 'y', 'z'
        
        Returns:
            Distance normalisée entre 0 et 1
        """
        if len(landmarks1) != len(landmarks2):
            return 1.0  # Distance maximale si longueurs différentes
        
        total_distance = 0.0
        count = 0
        
        for lm1, lm2 in zip(landmarks1, landmarks2):
            # Distance euclidienne en 3D
            dx = lm1['x'] - lm2['x']
            dy = lm1['y'] - lm2['y']
            dz = lm1['z'] - lm2['z']
            distance = (dx**2 + dy**2 + dz**2)**0.5
            total_distance += distance
            count += 1
        
        if count == 0:
            return 1.0
        
        # Normaliser par le nombre de landmarks
        return total_distance / count
    
    def recognize(self, landmarks: List[Dict], threshold: float = CONFIDENCE_THRESHOLD) -> tuple:
        """Reconnaître un signe à partir des landmarks.
        
        Args:
            landmarks: Liste des landmarks détectés
            threshold: Seuil de confiance minimum
        
        Returns:
            (label, confidence) ou ("", 0.0) si aucun signe reconnu
        """
        if not self.signs_cache:
            logging.warning(f"[SIGNS] Aucun signe chargé en mémoire!")
            return "", 0.0
        
        if not landmarks:
            logging.warning(f"[SIGNS] Aucun landmark fourni")
            return "", 0.0
        
        logging.info(f"[SIGNS] Reconnaissance: {len(landmarks)} landmarks vs {len(self.signs_cache)} signes en mémoire")
        
        best_match = None
        best_distance = float('inf')
        distances = {}
        
        for sign in self.signs_cache:
            stored_landmarks = sign.get('landmarks_json', [])
            if not stored_landmarks:
                continue
            
            # Calculer la distance
            distance = self._calculate_distance(landmarks, stored_landmarks)
            distances[sign.get('label', 'N/A')] = distance
            
            if distance < best_distance:
                best_distance = distance
                best_match = sign
        
        if best_match is None:
            logging.warning(f"[SIGNS] Aucun match trouvé (distances vides)")
            return "", 0.0
        
        # Convertir la distance en confidence (plus la distance est petite, plus la confidence est élevée)
        confidence = max(0.0, 1.0 - best_distance)
        
        logging.info(f"[SIGNS] Top 3: {sorted(distances.items(), key=lambda x: x[1])[:3]}")
        logging.info(f"[SIGNS] Meilleur match: {best_match.get('label', 'N/A')} (dist={best_distance:.4f}, conf={confidence:.4f}, threshold={threshold})")
        
        if confidence >= threshold:
            logging.info(f"[SIGNS] ✓ RECONNU: {best_match.get('label', 'N/A')} (conf={confidence:.4f})")
            return best_match['label'], confidence
        else:
            logging.info(f"[SIGNS] ✗ Confiance insuffisante: {confidence:.4f} < {threshold}")
        
        return "", 0.0
    
    def reload_signs(self):
        """Recharger les signes depuis la base de données."""
        self._load_signs()


# Instance globale du recognizer
sign_recognizer = SignRecognizer()

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
        logging.info("[CAMERA] Reader loop démarré")
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
                    
                    if frame_count % 30 == 0:  # Log tous les 30 frames
                        logging.info(f"[CAMERA] Frame {frame_count} lue")
                    
                    # === Détection MediaPipe (non-blocking) ===
                    if hand_detector.is_ready():
                        try:
                            detection_result = hand_detector.detect(frame, frame_count)
                            with self.lock:
                                self.last_detection = detection_result
                            if detection_result.get('num_hands', 0) > 0:
                                print(f"\n🎬 [CAMERA] Frame {frame_count}: {detection_result.get('num_hands', 0)} main(s) détectée(s)")
                                logging.info(f"[DETECTION] {detection_result.get('num_hands', 0)} main(s) frame {frame_count}")
                        except Exception as e:
                            logging.warning(f"[DETECTION_ERROR] {e}")
                    
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
            import copy
            return (self.last_frame.copy(), self.last_frame_time, copy.deepcopy(self.last_detection))
    
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
    logging.info(f"[STARTUP] Initialisation...")
    
    # Initialiser MediaPipe HandLandmarker
    if not hand_detector.is_ready():
        logging.warning("HandLandmarker non initialisé: %s", hand_detector.init_error)
    else:
        logging.info("[STARTUP] MediaPipe HandLandmarker prêt")
    
    # Initialiser la caméra
    if not camera_manager.init():
        logging.error("ERREUR: Impossible d'initialiser la caméra")
        logging.error("Message: %s", camera_manager.init_error)
    else:
        logging.info("[STARTUP] Caméra initialisée")
    
    # Vérifier les signes
    logging.info(f"[STARTUP] Signes chargés: {len(sign_recognizer.signs_cache)}")
    
    # Vérifier les signes
    logging.info(f"[SIGNS] {len(sign_recognizer.signs_cache)} signes chargés")


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
        "signs_loaded": len(sign_recognizer.signs_cache),
    }

# ============================================================================
# API GESTION DES SIGNES
# ============================================================================

from pydantic import BaseModel
from typing import Any, Optional

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
    """Lister tous les signes dans la base de données."""
    return {'signs': signs_db.get_all_signs()}

@app.get('/api/signs/{sign_id}')
def api_get_sign(sign_id: int):
    """Récupérer un signe spécifique."""
    s = signs_db.get_sign(sign_id)
    if not s:
        return {'error': 'not_found'}
    return {'sign': s}

@app.post('/api/signs')
def api_create_sign(payload: SignCreate):
    """Créer un nouveau signe."""
    sid = signs_db.create_sign(
        label=payload.label, 
        aliases=payload.aliases, 
        landmarks=payload.landmarks, 
        metadata=payload.metadata
    )
    # Recharger les signes dans le recognizer
    sign_recognizer.reload_signs()
    return {'id': sid}

@app.put('/api/signs/{sign_id}')
def api_update_sign(sign_id: int, payload: SignUpdate):
    """Mettre à jour un signe existant."""
    ok = signs_db.update_sign(
        sign_id, 
        label=payload.label, 
        aliases=payload.aliases, 
        landmarks=payload.landmarks, 
        metadata=payload.metadata
    )
    if ok:
        sign_recognizer.reload_signs()
    return {'ok': bool(ok)}

@app.delete('/api/signs/{sign_id}')
def api_delete_sign(sign_id: int):
    """Supprimer un signe."""
    ok = signs_db.delete_sign(sign_id)
    if ok:
        sign_recognizer.reload_signs()
    return {'ok': bool(ok)}

@app.get('/api/signs/export')
def api_export_signs():
    """Exporter tous les signes."""
    return {'signs': signs_db.export_all()}

@app.post('/api/signs/import')
def api_import_signs(payload: dict):
    """Importer une liste de signes."""
    signs = payload.get('signs') or []
    replace = bool(payload.get('replace'))
    inserted = signs_db.import_list(signs, replace=replace)
    sign_recognizer.reload_signs()
    return {'inserted': inserted}

@app.post('/api/signs/reload')
def api_reload_signs():
    """Recharger les signes depuis la base de données."""
    sign_recognizer.reload_signs()
    return {'signs_loaded': len(sign_recognizer.signs_cache)}

# ============================================================================
# WEBSOCKET
# ============================================================================

@app.websocket("/ws")
async def websocket_camera(ws: WebSocket):
    """WebSocket passif: lire les frames disponibles et envoyer les résultats.
    
    IMPORTANT:
    - NE PAS ouvrir/fermer la caméra ici
    - Lire uniquement la dernière frame disponible
    - Envoyer détections MediaPipe au client via WebSocket
    - Reconnaître les signes et envoyer le texte correspondant
    """
    print("[WEBSOCKET] === Handler WebSocket appelé ===")
    await ws.accept()
    print("[WEBSOCKET] Client accepté!")
    logging.info("[WEBSOCKET] Client connecté, statut caméra: {}, signes: {}".format(camera_manager.is_ready(), len(sign_recognizer.signs_cache)))
    
    try:
        logging.info("[WEBSOCKET] Préparation variables")
        # Variables locales pour cette connexion
        last_send_ts = 0.0
        last_detection_sent = {}
        
        # Buffers pour la reconnaissance de signes
        seq_buffer = deque(maxlen=SEQ_LEN)  # Buffer pour séquences de landmarks
        text_buffer = deque(maxlen=BUFFER_MAXLEN)  # Buffer pour stabilité du texte
        last_sent_text = ""
        last_sent_text_ts = 0.0
        
        # Envoyer un message d'accueil
        logging.info("[WEBSOCKET] Envoi message accueil")
        msg = {
            "status": "connected",
            "camera_ready": camera_manager.is_ready(),
            "mediapipe_ready": hand_detector.is_ready(),
            "signs_loaded": len(sign_recognizer.signs_cache),
            "message": "Connecté au serveur, en attente de frames"
        }
        logging.info(f"[WEBSOCKET] Message: {msg}")
        await ws.send_json(msg)
        logging.info("[WEBSOCKET] Message accueil envoyé")
        
        # Boucle principale: lire les frames et détections
        frame_skip = 0
        while True:
            # Vérifier si la caméra est prête
            if not camera_manager.is_ready():
                await asyncio.sleep(0.1)
                continue
            
            # Lire la dernière frame disponible (non-blocking)
            frame_data = camera_manager.read_frame()
            if frame_data is None:
                await asyncio.sleep(0.05)
                continue
            
            frame, frame_time, detection = frame_data
            
            # Throttle: traiter une frame sur N pour réduire charge CPU
            frame_skip += 1
            if frame_skip < 2:  # traiter 1 frame sur 2 (~15 FPS)
                continue
            frame_skip = 0
            
            now = time.time()
            
                # Traitement de la reconnaissance de signes
            recognized_text = ""
            confidence = 0.0
            
            if detection.get("num_hands", 0) > 0:
                # Prendre la première main détectée
                first_hand = detection["hands"][0]
                landmarks = first_hand["landmarks"]
                
                logging.info(f"[DETECTION] Main détectée: {len(landmarks)} landmarks")
                
                # Ajouter à la séquence
                seq_buffer.append(landmarks)
                
                # Reconnaître le signe si on a assez de frames
                if len(seq_buffer) >= 3:
                    logging.info(f"[RECOGNITION] Tentative reconnaissance avec {len(seq_buffer)} frames")
                    try:
                        print(f"  🔍 Reconnaissance du signe avec {len(seq_buffer)} frames de landmarks...")
                        recognized_text, confidence = sign_recognizer.recognize(list(seq_buffer))
                        
                        if recognized_text and confidence >= CONFIDENCE_THRESHOLD:
                            # Vérifier la stabilité
                            text_buffer.append((recognized_text, now))
                            
                            # Compter les occurrences récentes
                            cutoff = now - STABILITY_WINDOW
                            count = sum(1 for t, ts in text_buffer if ts >= cutoff and t == recognized_text)
                            
                            if count >= STABILITY_N:
                                # Signe stable détecté
                                if (recognized_text != last_sent_text or 
                                    (now - last_sent_text_ts) > REPETITION_COOLDOWN):
                                    
                                    logging.info(f"[SEND_SIGN] Envoi: {recognized_text} (conf={confidence:.2f})")
                                    
                                    # Envoyer le texte reconnu
                                    text_message = {
                                        "type": "recognized_text",
                                        "text": recognized_text,
                                        "confidence": confidence,
                                        "frames_used": count,
                                        "timestamp": datetime.now(timezone.utc).isoformat()
                                    }
                                    
                                    try:
                                        await ws.send_json(text_message)
                                        last_sent_text = recognized_text
                                        last_sent_text_ts = now
                                        logging.info(f"[RECOGNITION] Signe envoyé: {recognized_text} (conf: {confidence:.2f})")
                                    except Exception as e:
                                        logging.warning(f"[WEBSOCKET_ERROR] Erreur envoi: {e}")
                                        break
                    except Exception as e:
                        logging.exception(f"[RECOGNITION_ERROR] Erreur: {e}")
            
            # Envoyer les résultats MediaPipe
            if now - last_send_ts >= 0.1:  # ~10 Hz pour WebSocket
                # Construire le message avec détection
                message = {
                    "type": "detection",
                    "frames_total": camera_manager.get_frame_count(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "detection": detection
                }
                
                # Ajouter les infos de reconnaissance si disponible
                if recognized_text:
                    message["current_recognition"] = {
                        "text": recognized_text,
                        "confidence": confidence
                    }
                
                # Envoyer au client
                try:
                    await ws.send_json(message)
                except Exception as e:
                    logging.error(f"[WEBSOCKET_ERROR] Erreur envoi detection: {type(e).__name__}: {e}")
                    raise
                
                last_send_ts = now
            
            # Courte pause pour ne pas bloquer
            await asyncio.sleep(0.01)
    
    except WebSocketDisconnect:
        logging.info("[WEBSOCKET] Client déconnecté")
    except Exception as e:
        logging.exception(f"[WEBSOCKET_ERROR] Exception: {e}")
    finally:
        logging.info("[WEBSOCKET] Fermeture")


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
