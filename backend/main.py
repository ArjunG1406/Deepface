import asyncio
import base64
import json
import logging
import threading
import time
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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ---------- CONFIG (mirrors FaceDetectionApp.py) ----------
DETECTOR_BACKEND = "ssd"
MIN_FACE_SIZE    = 30
SMOOTH_N         = 5
TARGET_DELAY     = 0.08   # ~12 FPS analysis, same as desktop app


# ---------- HELPERS ----------

class Smoother:
    def __init__(self):
        self.history = deque(maxlen=SMOOTH_N)

    def update(self, label):
        self.history.append(label)
        return Counter(self.history).most_common(1)[0][0]


def to_python(obj):
    if isinstance(obj, dict):  return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [to_python(i) for i in obj]
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32,  np.int64)):    return int(obj)
    return obj


def decode_and_enhance(data_url):
    """Decode base64 frame and apply CLAHE enhancement — same as desktop app."""
    _, encoded = data_url.split(",", 1)
    arr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    # CLAHE on L channel — better emotion detection in low/uneven light
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def run_deepface(frame):
    """Run DeepFace with emotion + gender — no age to keep it fast."""
    try:
        results = DeepFace.analyze(
            frame,
            actions=["emotion", "gender"],
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
                    "box":     {"x": int(r["x"]), "y": int(r["y"]),
                                "w": int(r["w"]), "h": int(r["h"])},
                    "emotion": str(p["dominant_emotion"]),
                    "scores":  {k: float(v) for k, v in p.get("emotion", {}).items()},
                    "gender":  str(p.get("dominant_gender", p.get("gender", ""))),
                })
        return faces
    except Exception as e:
        logger.warning(f"DeepFace error: {e}")
        return []


# ---------- PER-CONNECTION ANALYSIS WORKER ----------
#
# Mirrors FaceDetectionApp.py's DetectionApp class:
#   - latest_frame  : written by WS receiver, read by analysis thread
#   - current_faces : written by analysis thread, read by WS sender
# Both protected by threading.Lock() to avoid race conditions.

class AnalysisWorker:
    def __init__(self):
        self.latest_frame  = None
        self.current_faces = []
        self.smoothers     = [Smoother() for _ in range(5)]
        self.frame_lock    = threading.Lock()
        self.face_lock     = threading.Lock()
        self.running       = True

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def push_frame(self, frame):
        """Called by WS receiver whenever a new frame arrives."""
        with self.frame_lock:
            self.latest_frame = frame

    def get_faces(self):
        """Called by WS sender to get the latest smoothed results."""
        with self.face_lock:
            return list(self.current_faces)

    def stop(self):
        self.running = False

    def _loop(self):
        """
        Dedicated analysis thread — exactly like _detection_loop() in
        FaceDetectionApp.py. Runs DeepFace at ~12fps independently of
        how fast frames arrive from the browser.
        """
        last_time = 0
        while self.running:
            now = time.time()
            if now - last_time < TARGET_DELAY:
                time.sleep(0.005)
                continue
            last_time = now

            with self.frame_lock:
                if self.latest_frame is None:
                    continue
                frame = self.latest_frame.copy()

            faces = run_deepface(frame)

            valid = []
            for i, face in enumerate(faces):
                s = self.smoothers[i] if i < 5 else Smoother()
                valid.append({
                    "box":     face["box"],
                    "emotion": s.update(face["emotion"]),
                    "scores":  face["scores"],
                    "gender":  face["gender"],
                })

            with self.face_lock:
                self.current_faces = valid


# ---------- WEBSOCKET ENDPOINT ----------

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")
    worker = AnalysisWorker()

    try:
        while True:
            data = await websocket.receive_text()
            fd = json.loads(data).get("frame")
            if not fd:
                continue

            # Decode + enhance frame and hand it to the analysis thread
            frame = decode_and_enhance(fd)
            if frame is None:
                continue
            worker.push_frame(frame)

            # Send whatever the analysis thread has computed so far
            faces = worker.get_faces()
            await websocket.send_json({
                "faces": to_python(faces),
                "count": len(faces)
            })

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WS error: {e}")
    finally:
        worker.stop()


@app.get("/health")
def health():
    return {"status": "ok"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")