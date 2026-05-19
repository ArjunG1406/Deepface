import sys
import cv2
import threading
from collections import deque, Counter
from deepface import DeepFace

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QPushButton, QFrame, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap

# ---------- CONFIG ----------
DETECTOR_BACKEND = "ssd"
MIN_FACE_SIZE    = 30
SMOOTH_N         = 5

EMOTION_COLORS = {
    "happy":    (255, 229, 0),
    "sad":      (255, 140, 120),
    "angry":    (60,  60,  255),
    "fear":     (0,   153, 255),
    "surprise": (255, 228, 77),
    "disgust":  (0,   255, 157),
    "neutral":  (90,  96,  112),
}
EMOTION_HEX = {
    "happy":    "#00e5ff",
    "sad":      "#788cff",
    "angry":    "#ff3cac",
    "fear":     "#ff9900",
    "surprise": "#ffe44d",
    "disgust":  "#00ff9d",
    "neutral":  "#8899aa",
}


class Smoother:
    def __init__(self):
        self.history = deque(maxlen=SMOOTH_N)
    def update(self, label):
        self.history.append(label)
        return Counter(self.history).most_common(1)[0][0]


class Signals(QObject):
    frame_ready = pyqtSignal(object)


class DetectionApp:
    """
    Two separate threads:
    1. Camera thread  — reads frames at full speed (~30fps), draws boxes, emits frames
    2. Analysis thread — runs DeepFace on latest frame, updates face data
    This way the video is always smooth even when DeepFace is slow.
    """
    def __init__(self, signals):
        self.signals       = signals
        self.running       = False
        self.latest_frame  = None
        self.current_faces = []
        self.smoothers     = [Smoother() for _ in range(5)]
        self.frame_lock    = threading.Lock()
        self.face_lock     = threading.Lock()

    def start(self):
        self.running = True
        # Camera thread — runs at full fps
        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()
        # Analysis thread — runs DeepFace as fast as it can
        self.det_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.det_thread.start()

    def stop(self):
        self.running = False

    def _camera_loop(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Enhance frame for better detection in low light
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

            # Store latest frame for analysis thread
            with self.frame_lock:
                self.latest_frame = enhanced.copy()

            # Draw current faces on display frame
            display = cv2.flip(frame, 1)
            h, w = display.shape[:2]

            with self.face_lock:
                faces = list(self.current_faces)

            for face in faces:
                fx, fy, fw, fh = face["box"]
                fx = w - fx - fw  # mirror x
                emotion = face["emotion"]
                color   = EMOTION_COLORS.get(emotion, (0, 229, 255))

                # Box
                cv2.rectangle(display, (fx, fy), (fx+fw, fy+fh), color, 2)
                # Emotion label above
                cv2.rectangle(display, (fx, fy-25), (fx+fw, fy), color, -1)
                cv2.putText(display, emotion.capitalize(), (fx+5, fy-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                # Age + gender below
                age    = face.get("age", "")
                gender = face.get("gender", "")
                g_sym  = "M" if gender == "Man" else "F" if gender == "Woman" else ""
                info   = f"{g_sym} {age}yrs" if age else ""
                if info:
                    (iw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(display, (fx, fy+fh+1), (fx+iw+10, fy+fh+22), color, -1)
                    cv2.putText(display, info, (fx+5, fy+fh+16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            self.signals.frame_ready.emit((display, faces))

        cap.release()

    def _detection_loop(self):
        """Runs DeepFace at controlled FPS for smoother UI."""
        import time
        last_time = 0
        target_delay = 0.08   # ~12 FPS detection (good balance)

        while self.running:
            now = time.time()
            if now - last_time < target_delay:
                time.sleep(0.005)
                continue
            last_time = now

            with self.frame_lock:
                if self.latest_frame is None:
                    continue
                frame = self.latest_frame.copy()   # safer copy

            try:
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion", "age", "gender"],
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    silent=True
                )

                if isinstance(results, dict):
                    results = [results]

                valid = []
                for i, p in enumerate(results):
                    r = p["region"]
                    if r["w"] >= MIN_FACE_SIZE and r["h"] >= MIN_FACE_SIZE:
                        s = self.smoothers[i] if i < 5 else Smoother()
                        valid.append({
                            "box": (r["x"], r["y"], r["w"], r["h"]),
                            "emotion": s.update(str(p["dominant_emotion"])),
                            "scores": {k: float(v) for k, v in p.get("emotion", {}).items()},
                            "age": max(1, int(p.get("age", 0)) - 10),
                            "gender": str(p.get("dominant_gender", p.get("gender", ""))),
                        })

                with self.face_lock:
                    self.current_faces = valid

            except Exception as e:
                pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepFace Emotion Detection")
        self.setMinimumSize(900, 580)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #050810; color: #e8eaf0; }
            QLabel { color: #e8eaf0; }
            QPushButton {
                font-size: 12px; padding: 10px 24px;
                border: 1px solid #00e5ff; border-radius: 3px;
                color: #00e5ff; background: transparent;
            }
            QPushButton:hover { background-color: #00e5ff; color: #050810; }
            QPushButton:disabled { border-color: #5a6070; color: #5a6070; }
            QPushButton#stopBtn { border-color: #ff3cac; color: #ff3cac; }
            QPushButton#stopBtn:hover { background-color: #ff3cac; color: #050810; }
            QFrame#panel {
                background-color: #0d1120;
                border: 1px solid rgba(255,255,255,0.07);
                border-radius: 4px;
            }
        """)

        self.signals = Signals()
        self.signals.frame_ready.connect(self.update_frame)
        self.app     = None
        self.fps_count = 0
        self.frame_total = 0
        self._build_ui()

        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self._tick_fps)
        self.fps_timer.start(1000)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(20)

        # Left — video
        left = QVBoxLayout(); left.setSpacing(10)
        lbl = QLabel("// LIVE FEED")
        lbl.setStyleSheet("color:#5a6070; font-size:10px; letter-spacing:2px;")
        left.addWidget(lbl)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#0d1120; border:1px solid rgba(255,255,255,0.07); border-radius:4px; color:#5a6070; font-size:14px;")
        self.video_label.setText("📷   Click Start Detection")
        left.addWidget(self.video_label)

        btns = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start Detection")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        btns.addStretch()
        left.addLayout(btns)
        root.addLayout(left, 3)

        # Right — sidebar
        right = QVBoxLayout(); right.setSpacing(14); right.setAlignment(Qt.AlignTop)

        self.status_lbl = QLabel("● IDLE")
        self.status_lbl.setStyleSheet("color:#5a6070; font-size:11px; letter-spacing:2px;")
        right.addWidget(self.status_lbl)

        stats = QFrame(); stats.setObjectName("panel")
        sl = QVBoxLayout(stats)
        t = QLabel("// STATS"); t.setStyleSheet("color:#5a6070; font-size:10px; letter-spacing:2px; margin-bottom:6px;")
        sl.addWidget(t)
        grid = QGridLayout(); grid.setSpacing(8)
        self.stat_faces,  self.val_faces  = self._stat("0", "Faces")
        self.stat_fps,    self.val_fps    = self._stat("—", "FPS")
        self.stat_frames, self.val_frames = self._stat("0", "Frames")
        grid.addWidget(self.stat_faces, 0, 0)
        grid.addWidget(self.stat_fps,   0, 1)
        grid.addWidget(self.stat_frames,1, 0)
        sl.addLayout(grid)
        right.addWidget(stats)

        det = QFrame(); det.setObjectName("panel")
        det.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        dl = QVBoxLayout(det)
        dt = QLabel("// DETECTIONS"); dt.setStyleSheet("color:#5a6070; font-size:10px; letter-spacing:2px; margin-bottom:6px;")
        dl.addWidget(dt)
        self.det_lbl = QLabel("No faces detected")
        self.det_lbl.setStyleSheet("color:#5a6070; font-size:11px;")
        self.det_lbl.setWordWrap(True)
        self.det_lbl.setAlignment(Qt.AlignTop)
        dl.addWidget(self.det_lbl)
        right.addWidget(det)
        root.addLayout(right, 1)

    def _stat(self, val, label):
        f = QFrame()
        f.setStyleSheet("background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:3px;")
        l = QVBoxLayout(f); l.setContentsMargins(10, 8, 10, 8)
        v = QLabel(val); v.setStyleSheet("color:#00e5ff; font-size:26px; font-weight:bold;")
        lb = QLabel(label.upper()); lb.setStyleSheet("color:#5a6070; font-size:9px; letter-spacing:1px;")
        l.addWidget(v); l.addWidget(lb)
        return f, v

    def start(self):
        self.app = DetectionApp(self.signals)
        self.app.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_lbl.setText("● LIVE")
        self.status_lbl.setStyleSheet("color:#00ff9d; font-size:11px; letter-spacing:2px;")

    def stop(self):
        if self.app:
            self.app.stop()
            self.app = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_lbl.setText("● IDLE")
        self.status_lbl.setStyleSheet("color:#5a6070; font-size:11px; letter-spacing:2px;")
        self.video_label.clear()
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("📷   Click Start Detection")
        self.det_lbl.setText("No faces detected")
        self.val_faces.setText("0")
        self.val_fps.setText("—")
        self.val_frames.setText("0")
        self.frame_total = 0
        self.fps_count   = 0

    def update_frame(self, data):
        display, faces = data
        self.fps_count  += 1
        self.frame_total += 1
        self.val_frames.setText(str(self.frame_total))
        self.val_faces.setText(str(len(faces)))

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, w*c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

        if not faces:
            self.det_lbl.setText("No faces detected")
            return
        text = ""
        for i, face in enumerate(faces):
            color = EMOTION_HEX.get(face["emotion"], "#00e5ff")
            g_icon = "♂" if face.get("gender") == "Man" else "♀" if face.get("gender") == "Woman" else ""
            text += f"<b style='color:{color}'>Face #{i+1} — {face['emotion'].capitalize()}</b><br>"
            text += f"<span style='color:#aabbcc'>{g_icon} {face.get('gender','—')}  |  Age: {face.get('age','—')}</span><br>"
            scores = sorted(face.get("scores", {}).items(), key=lambda x: x[1], reverse=True)
            for emo, val in scores[:3]:
                text += f"<span style='color:#5a6070'>{emo}: {val:.0f}%</span><br>"
            text += "<br>"
        self.det_lbl.setText(text)
        self.det_lbl.setTextFormat(Qt.RichText)

    def _tick_fps(self):
        self.val_fps.setText(str(self.fps_count))
        self.fps_count = 0

    def closeEvent(self, event):
        self.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())