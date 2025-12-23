from fastapi import FastAPI, WebSocket
import cv2
import mediapipe as mp
from tts import speak

app = FastAPI()
mp_hands = mp.solutions.hands.Hands()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # TODO: landmarks + model inference
        text = "SIGN_DETECTED"

        await ws.send_json({"text": text})

        # Optional TTS
        # speak(text)
