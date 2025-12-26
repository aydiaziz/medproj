# 🤖 Détecteur de Signes — Sign Recognition System

> **Real-time hand gesture detection and recognition using MediaPipe Hands, FastAPI, and WebSocket**

Un système complet de détection de signes en temps réel avec interface web moderne et architecture serveur centralisée.

---

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Usage](#usage)
5. [API Reference](#api-reference)
6. [Structure du projet](#structure-du-projet)
7. [Technologies](#technologies)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Vue d'ensemble

Ce projet implémente une **architecture centralisée pour la détection de mains en temps réel** :

### Caractéristiques principales :

- ✅ **Caméra unique et persistante** : une seule ouverture au démarrage, fermeture à l'arrêt
- ✅ **Thread dédié** : lecture continue des frames en background
- ✅ **MediaPipe Hands** : détection des 21 landmarks de chaque main
- ✅ **WebSocket live** : streaming des détections au navigateur (~10 Hz)
- ✅ **Interface web** : affichage des landmarks en grille responsive
- ✅ **Thread-safe** : utilisation de locks pour accès concurrent

### Data Flow :

```
┌──────────────────────────────────────────────────────────────┐
│                      Architecture                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [STARTUP]                                                  │
│      ↓                                                       │
│  • Ouvrir caméra (une fois)                                │
│  • Initialiser MediaPipe HandLandmarker                    │
│  • Démarrer camera_reader_thread                           │
│                                                              │
│  ┌─────────────────────────────────────────────────┐       │
│  │     CAMERA READER THREAD (continu)              │       │
│  │  ┌──────────────┐      ┌──────────────┐        │       │
│  │  │ cv2.read()   │──→   │ MediaPipe    │──→     │       │
│  │  │ (30 FPS)     │      │ detect()     │  last_ │       │
│  │  │ frame = F    │      │ detection    │  frame │       │
│  │  └──────────────┘      └──────────────┘        │       │
│  └─────────────────────────────────────────────────┘       │
│            ↓                          ↓                     │
│       [Lock]                    [Lock]                      │
│            ↓                          ↓                     │
│  last_frame ←─────────────────────────→ last_detection    │
│                                                              │
│  ┌─────────────────────────────────────────────────┐       │
│  │         WEBSOCKET HANDLER (per-client)          │       │
│  │  • Lire last_frame (pas de gestion)             │       │
│  │  • Lire last_detection                          │       │
│  │  • Envoyer au client via JSON (~10 Hz)         │       │
│  └─────────────────────────────────────────────────┘       │
│            ↓                                                │
│       [Network]                                            │
│            ↓                                                │
│  ┌─────────────────────────────────────────────────┐       │
│  │         FRONTEND (Browser)                      │       │
│  │  • Afficher count de mains                      │       │
│  │  • Grille de 21 landmarks par main             │       │
│  │  • Coords (x, y, z) + confiance               │       │
│  └─────────────────────────────────────────────────┘       │
│                                                              │
│  [SHUTDOWN]                                                 │
│      ↓                                                       │
│  • Arrêter camera_reader_thread                           │
│  • Fermer caméra (une fois)                               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

### Backend (`backend/app_new.py`)

#### 1. **CameraManager** — Gestion centralisée de la caméra

```python
camera_manager = CameraManager(device=0, width=640, height=480)

# À startup
camera_manager.init()  # Ouvre caméra UNE FOIS, démarre thread

# Dans le thread dédié
def _reader_loop():
    while running:
        ret, frame = cap.read()  # Lecture continue
        if ret:
            last_frame = frame.copy()
            detection = hand_detector.detect(frame, frame_count)
            last_detection = detection

# À shutdown
camera_manager.shutdown()  # Ferme caméra, arrête thread
```

**Avantages** :
- Évite les réouvertures caméra (qui causent des `ret=False`)
- Une seule instance partagée
- Thread-safe avec `threading.RLock()`
- WebSocket lit uniquement, ne gère jamais la caméra

#### 2. **HandLandmarkerWrapper** — Détection MediaPipe

```python
class HandLandmarkerWrapper:
    def detect(frame: np.ndarray, frame_count: int) -> dict:
        """Détecte les mains dans la frame."""
        result = self.landmarker.detect_for_video(mp_image, frame_count)
        # Retourne:
        {
            "num_hands": 2,
            "hands": [
                {
                    "index": 0,
                    "label": "Left",
                    "confidence": 0.95,
                    "landmarks": [
                        {"x": 0.45, "y": 0.38, "z": -0.08, "presence": 1.0},
                        ...21 landmarks...
                    ]
                },
                ...
            ]
        }
```

**Points clés** :
- Utilise MediaPipe Tasks API (VIDEO mode)
- Détecte jusqu'à 2 mains simultanément
- Landmarks : 21 points par main (x, y, z, présence)
- Fallback automatique IMAGE mode si VIDEO échoue

#### 3. **WebSocket Handler** — Streaming live

```python
@app.websocket("/ws")
async def websocket_camera(ws: WebSocket):
    # Accepte la connexion
    await ws.accept()
    
    # Boucle principale : lire et envoyer
    while True:
        # Lire dernière frame (non-blocking)
        frame_data = camera_manager.read_frame()
        
        # Construire message
        message = {
            "type": "detection",
            "frames_total": 150,
            "timestamp": "2025-12-25T14:30:45...",
            "detection": {...}
        }
        
        # Envoyer au client
        await ws.send_json(message)  # ~10 Hz
```

### Frontend (`frontend/index.html`)

#### 1. **WebSocket Client** — Connexion au serveur

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);
    updateDetection(data);  // Afficher landmarks
};
```

#### 2. **Detection Display** — Rendu des mains

```javascript
function renderHandLandmarks(hand) {
    // Pour chaque main:
    // - Label (Left/Right) + Confiance
    // - Grille de 21 landmarks
    // - Coords (x, y, z) arrondies
}
```

---

## 📦 Installation

### 1. Prérequis

- **Python 3.11** (recommandé)
- **Caméra USB** (webcam)
- **Navigateur moderne** (Chrome, Firefox, Edge)

### 2. Setup venv

```bash
# Créer venv
python -m venv .venv311

# Activer
.venv311\Scripts\activate  # Windows
source .venv311/bin/activate  # Linux/Mac

# Installer dépendances
pip install -r requirements.txt
```

### 3. Modèle MediaPipe

Le fichier `backend/models/hand_landmarker.task` est nécessaire. S'il manque :

```bash
mkdir -p backend/models
cd backend/models
# Le serveur tentera de télécharger depuis Google si absent
```

---

## 🚀 Usage

### Démarrer le serveur

```bash
cd medproj
.venv311\Scripts\python.exe -m uvicorn backend.app_new:app --host 127.0.0.1 --port 8000
```

**Output** :
```
INFO:     Started server process [1234]
INFO:     Application startup complete
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Accéder au frontend

Ouvrir navigateur :
```
http://127.0.0.1:8000
```

### Utiliser l'interface

1. **"Démarrer détection"** : lance la détection
2. **Placer mains devant caméra** : landmarks s'affichent
3. **"Arrêter détection"** : pause l'affichage

### Logs du serveur

```
INFO:     127.0.0.1:60267 - "WebSocket /ws" [accepted]
INFO:     connection open
WARNING:root:Erreur détection mains: ... (si capteur occulté)
```

---

## 📡 API Reference

### HTTP Endpoints

#### `GET /`
Retourne la page HTML du frontend.

```
curl http://127.0.0.1:8000/
→ <html>...</html>
```

#### `GET /health`
Status de l'application.

```
curl http://127.0.0.1:8000/health
→ {
    "status": "ok",
    "camera_ready": true,
    "frames_read": 150
  }
```

### WebSocket Endpoint

#### `WS /ws`
Streaming des détections MediaPipe.

**Message Client → Serveur** : (none — just websocket, no JSON client → server)

**Message Serveur → Client** :
```json
{
  "type": "detection",
  "frames_total": 150,
  "timestamp": "2025-12-25T14:30:45.123456+00:00",
  "detection": {
    "num_hands": 2,
    "hands": [
      {
        "index": 0,
        "label": "Left",
        "confidence": 0.952,
        "landmarks": [
          {
            "x": 0.4513,
            "y": 0.3847,
            "z": -0.0827,
            "presence": 1.0
          },
          ...21 landmarks total...
        ]
      },
      {
        "index": 1,
        "label": "Right",
        "confidence": 0.948,
        "landmarks": [...]
      }
    ]
  }
}
```

**Fréquence** : ~10 Hz (1 message toutes les 100 ms)

**Landmarks** : 21 points par main (MediaPipe standard)
- 0: Wrist
- 1-4: Thumb
- 5-8: Index
- 9-12: Middle
- 13-16: Ring
- 17-20: Pinky

---

## 📁 Structure du projet

```
medproj/
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── backend/
│   ├── app_new.py                     # Serveur FastAPI (ACTIF)
│   ├── app.py                         # Ancien serveur (déprécié)
│   ├── tts.py                         # Module TTS (optionnel)
│   ├── signs_db.py                    # SQLite signs database
│   └── models/
│       └── hand_landmarker.task       # Modèle MediaPipe (téléchargé auto)
├── frontend/
│   └── index.html                     # Interface web (servie par /route)
└── .venv311/                          # Virtual environment Python 3.11
```

### Fichiers clés

| Fichier | Rôle |
|---------|------|
| `backend/app_new.py` | ⭐ Serveur principal (CameraManager + HandLandmarker + WebSocket) |
| `backend/app.py` | Ancien serveur (per-connection camera, deprecated) |
| `frontend/index.html` | Interface web (affichage landmarks, contrôles) |
| `requirements.txt` | Dépendances (`fastapi`, `uvicorn`, `opencv-python`, `mediapipe`, etc.) |

---

## 🛠️ Technologies

### Backend
- **FastAPI** : Framework web asynchrone
- **Uvicorn** : Serveur ASGI
- **OpenCV (cv2)** : Capture caméra
- **MediaPipe** : Détection mains (Tasks API)
- **Python 3.11** : Runtime

### Frontend
- **HTML5** : Markup
- **CSS3** : Styling moderne
- **JavaScript (vanilla)** : WebSocket client
- **getUserMedia API** : Aperçu caméra local (optionnel)

### Infrastructure
- **WebSocket** : Communication bidirectionnelle temps-réel
- **JSON** : Format échange données
- **Threading** : Concurrence Python
- **asyncio** : Programmation asynchrone FastAPI

---

## 🐛 Troubleshooting

### ❌ "Impossible d'ouvrir la caméra"

**Cause** : Caméra non connectée ou permissions refusées

**Solution** :
```python
# Vérifier caméra disponible
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"

# Essayer device 1, 2, etc.
# Dans app_new.py: CAMERA_DEVICE = 1
```

### ❌ "MediaPipe Tasks API non disponible"

**Cause** : Import échoue (version mediapipe incompatible)

**Solution** :
```bash
# Réinstaller mediapipe
pip uninstall mediapipe -y
pip install mediapipe==0.10.31
```

### ❌ "Erreur envoi WebSocket: ..."

**Cause** : Client déconnecté (normal à l'arrêt)

**Solution** : Rien — c'est attendu. Vérifier dans logs :
```
INFO:     127.0.0.1:60267 - "WebSocket /ws" [accepted]
INFO:     connection closed  # ← Normal
```

### ❌ Port 8000 déjà utilisé

**Cause** : Autre serveur en écoute

**Solution** :
```bash
# Tuer ancien processus
taskkill /F /IM python.exe  # Windows
lsof -i :8000 | grep -v PID | awk '{print $2}' | xargs kill -9  # Linux/Mac

# Ou utiliser autre port
uvicorn backend.app_new:app --host 127.0.0.1 --port 8001
```

### ❌ Landmarks vides ou "0 mains détectées"

**Cause** : Détecteur pas actif ou mains hors-champ

**Solution** :
1. Cliquer "▶ Démarrer détection"
2. Placer mains devant caméra (bien visibles)
3. Vérifier stats "Frames traitées" augmente
4. Vérifier WebSocket connecté ("Connecté ✓")

---

## 🎓 Concepts clés

### Pourquoi une caméra centralisée ?

**Problème (ancien app.py)** :
- Ouvrir caméra par WebSocket connection
- Si 2 clients → 2 appels `cv2.VideoCapture(0)`
- Résultat : `ret=False`, reopen loops, instabilité

**Solution (app_new.py)** :
- 1 caméra = 1 ouverture = 1 thread
- Thread lit en continu, stocke `last_frame`
- Tous les clients lisent `last_frame` (pas de réouverture)
- Robuste, scalable, thread-safe

### Pourquoi VIDEO mode MediaPipe ?

**IMAGE mode** :
- 1 frame = 1 détection indépendante
- Pas de contexte temporel

**VIDEO mode** :
- Utilise frames précédentes (contexte)
- Plus rapide, plus stable
- Meilleure détection sur mouvements

---

## 📈 Performance

### Metrics

| Métrique | Valeur |
|----------|--------|
| Résolution caméra | 640×480 (configurable) |
| FPS caméra | 30 (configurable) |
| FPS détection | ~15-20 (throttle) |
| Fréquence WebSocket | ~10 Hz (configurable) |
| Landmarks par main | 21 points |
| Mains simultanées | Jusqu'à 2 |
| Latence détection | ~30-50 ms |
| Bande passante WebSocket | ~5-10 KB/s |

### Optimisations

1. **Frame skipping** : traiter 1/2 frames (throttle CPU)
2. **Throttle WebSocket** : ~10 Hz (pas utile d'aller plus vite)
3. **Copy-on-read** : `frame.copy()` pour éviter race conditions
4. **Lock-free stats** : logging sans contention

---

## 🔮 Futures améliorations

- [ ] Reconnaissance gestes (LSTM/transformer)
- [ ] TTS French synth
- [ ] Base de données gestes
- [ ] Multi-client robustness (load balancing)
- [ ] WebRTC pour bandwidth reduit
- [ ] GPU acceleration (CUDA/OpenGL)
- [ ] Configuration UI (résolution, FPS, confiance)

---

## 📝 License

MIT (ou selon votre préférence)

## 👥 Auteur

Created with ❤️ for sign recognition research

---

**Last updated**: 2025-12-26
