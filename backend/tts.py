import threading
import pyttsx3

# Initialize engine once. Selecting a French voice if available.
engine = pyttsx3.init()
voices = engine.getProperty('voices') or []
selected_voice = None
for v in voices:
    name = (v.name or "").lower()
    vid = (v.id or "").lower()
    if "fr" in name or "french" in name or "fr" in vid:
        selected_voice = v.id
        break
if selected_voice:
    try:
        engine.setProperty('voice', selected_voice)
    except Exception:
        # best-effort: if it fails, continue with defaults
        pass

# set a friendly speaking rate for French
try:
    engine.setProperty('rate', 150)
except Exception:
    pass

# pyttsx3 is not async; run speak in a separate thread to avoid blocking
_tts_lock = threading.Lock()

def speak(text: str):
    """Speak text (blocking). Caller should call this within a thread if non-blocking behavior is required."""
    with _tts_lock:
        engine.say(text)
        engine.runAndWait()
