import asyncio
import base64
import json
import logging
from collections import deque, Counter

import cv2
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])




# Exact same settings as your original working script
DETECTOR_BACKEND = "ssd"
FRAME_SKIP       = 3
MIN_FACE_SIZE    = 30
SMOOTH_N         = 5


class Smoother:
    def __init__(self):
        self.history = deque(maxlen=SMOOTH_N)

    def update(self, label):
        self.history.append(label)
        return Counter(self.history).most_common(1)[0][0]


def decode_frame(data_url):
    _, encoded = data_url.split(",", 1)
    arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def to_python(obj):
    if isinstance(obj, dict):  return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [to_python(i) for i in obj]
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32,  np.int64)):    return int(obj)
    return obj


def analyze(frame):
    try:
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            silent=True
        )
        if isinstance(results, dict):
            results = [results]
        faces = []
        for p in results:
            r = p["region"]
            if r["w"] >= MIN_FACE_SIZE and r["h"] >= MIN_FACE_SIZE:
                faces.append({
                    "box":     {"x": int(r["x"]), "y": int(r["y"]), "w": int(r["w"]), "h": int(r["h"])},
                    "emotion": str(p["dominant_emotion"]),
                    "scores":  {k: float(v) for k, v in p.get("emotion", {}).items()},

                })
        return faces
    except Exception as e:
        logger.warning(f"DeepFace: {e}")
        return []


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")
    smoothers = [Smoother() for _ in range(5)]
    loop = asyncio.get_event_loop()
    frame_count = 0

    try:
        while True:
            data = await websocket.receive_text()
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue
            fd = json.loads(data).get("frame")
            if not fd:
                continue
            frame = decode_frame(fd)
            if frame is None:
                continue
            faces = await loop.run_in_executor(None, analyze, frame)
            out = []
            for i, face in enumerate(faces):
                s = smoothers[i] if i < 5 else Smoother()
                out.append({
                    "box":     face["box"],
                    "emotion": s.update(face["emotion"]),
                    "scores":  face["scores"],

                })
            await websocket.send_json({"faces": to_python(out), "count": len(out)})
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WS: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")