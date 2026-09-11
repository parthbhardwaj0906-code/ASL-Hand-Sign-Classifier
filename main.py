import os
from collections import deque

import av
import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------------------------------------------------------
# Absolute Path Resolution
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

MEDIA_STREAM_CONSTRAINTS = {
    "video": {
        "width": {"ideal": 640, "max": 640},
        "height": {"ideal": 480, "max": 480},
        "frameRate": {"ideal": 24, "max": 30},
    },
    "audio": False,
}

METERED_API_KEY = "ad5309263f48ad526cb68cac07785003e110"
METERED_USERNAME = "f7973cbc48d0dfdb72db313f"
METERED_CREDENTIAL = "GYCeNokosWrvSpfL"

@st.cache_data(ttl=3600)
def get_rtc_configuration():
    try:
        url = f"https://api.metered.ca/api/v1/turn/credentials?apiKey={METERED_API_KEY}"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            return RTCConfiguration({"iceServers": response.json()})
    except Exception:
        pass

    return RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
                {
                    "urls": "turn:global.relay.metered.ca:80",
                    "username": METERED_USERNAME,
                    "credential": METERED_CREDENTIAL,
                },
                {
                    "urls": "turn:global.relay.metered.ca:443",
                    "username": METERED_USERNAME,
                    "credential": METERED_CREDENTIAL,
                },
            ]
        }
    )

RTC_CONFIG = get_rtc_configuration()

# -----------------------------------------------------------------------------
# Sign Configuration & Landmark Links
# -----------------------------------------------------------------------------
TARGET_SIGNS = {
    "A": "Fist with thumb resting along the side of the index finger",
    "B": "Four fingers straight up and flat together, thumb tucked across palm",
    "C": "All fingers curved forming a wide C-shape",
    "D": "Index finger straight up, thumb touching tips of middle, ring, and pinky fingers",
    "I": "Pinky extended straight up, all other fingers folded in, thumb tucked",
    "L": "Index and thumb extended outward at 90 degrees",
    "O": "All fingertips curved touching thumb tip to form a circle",
    "U": "Index and middle fingers extended straight up and touching together",
    "V": "Index and middle fingers extended apart in a V-shape",
    "W": "Index, middle, and ring fingers extended upward",
}
TARGET_KEYS = list(TARGET_SIGNS.keys())

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# -----------------------------------------------------------------------------
# WebRTC Video Processor
# -----------------------------------------------------------------------------
class SignTrainerProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.detector = None
        self.target_idx = 0
        self.score_buffer = deque(maxlen=4)
        self.frame_counter = 0

    def _init_detector(self):
        """Lazy initialization inside the processor thread."""
        if self.detector is None:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Missing MediaPipe task model at: {MODEL_PATH}")
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)

    def set_target_idx(self, idx: int):
        self.target_idx = idx
        self.score_buffer.clear()

    def draw_skeleton(self, frame, landmarks):
        h, w, _ = frame.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(frame, points[p1], points[p2], (255, 255, 255), 2)

        for pt in points:
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)

    def calculate_finger_extensions(self, landmarks):
        wrist = np.array([landmarks[0].x, landmarks[0].y])
        finger_indices = [(4, 2), (8, 6), (12, 10), (16, 14), (20, 18)]

        extensions = []
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
        thumb_dist = np.linalg.norm(thumb_tip - wrist)
        extensions.append(np.clip((thumb_dist - 0.15) / 0.18, 0.0, 1.0))

        for tip_idx, pip_idx in finger_indices[1:]:
            tip, pip = landmarks[tip_idx], landmarks[pip_idx]
            diff = pip.y - tip.y
            extensions.append(np.clip((diff + 0.02) / 0.12, 0.0, 1.0))

        return extensions

    def evaluate_gesture(self, landmarks):
        target = TARGET_KEYS[self.target_idx]
        ext = self.calculate_finger_extensions(landmarks)

        wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
        index_mcp = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
        index_pip = np.array([landmarks[6].x, landmarks[6].y, landmarks[6].z])

        index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
        middle_tip = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
        ring_tip = np.array([landmarks[16].x, landmarks[16].y, landmarks[16].z])
        pinky_tip = np.array([landmarks[20].x, landmarks[20].y, landmarks[20].z])

        palm_size = np.linalg.norm(index_mcp - wrist)
        if palm_size == 0:
            palm_size = 0.20

        norm_thumb_to_mcp = np.linalg.norm(thumb_tip - index_mcp) / palm_size
        norm_thumb_to_pip = np.linalg.norm(thumb_tip - index_pip) / palm_size
        norm_thumb_to_pinky = np.linalg.norm(thumb_tip - pinky_tip) / palm_size

        score = 0.20

        if target == "A":
            if all(e < 0.35 for e in ext[1:]) and ((norm_thumb_to_mcp < 0.38) or (norm_thumb_to_pip < 0.38)):
                score = 0.95
            elif all(e < 0.35 for e in ext[1:]):
                score = 0.45
        elif target == "B":
            if all(e > 0.65 for e in ext[1:]) and (norm_thumb_to_pinky < 0.95 or norm_thumb_to_mcp < 0.50):
                score = 0.94
            elif all(e > 0.65 for e in ext[1:]):
                score = 0.45
        elif target == "C":
            d_index = np.linalg.norm(index_tip - wrist) / palm_size
            if all(1.0 < d < 2.1 for d in [d_index]):
                score = 0.94
        elif target == "D":
            if ext[1] > 0.60 and ext[2] < 0.45 and ext[3] < 0.45 and ext[4] < 0.45:
                score = 0.95
        elif target == "I":
            if ext[4] > 0.55 and all(e < 0.45 for e in ext[1:4]):
                score = 0.95
        elif target == "L":
            if ext[1] > 0.55 and all(e < 0.40 for e in ext[2:]) and norm_thumb_to_mcp > 0.85:
                score = 0.94
        elif target == "O":
            tips = [index_tip, middle_tip, ring_tip, pinky_tip]
            if np.mean([np.linalg.norm(thumb_tip - t) / palm_size for t in tips]) < 0.45:
                score = 0.93
        elif target == "U":
            if ext[1] > 0.60 and ext[2] > 0.60 and ext[3] < 0.40 and ext[4] < 0.40:
                score = 0.95
        elif target == "V":
            if ext[1] > 0.60 and ext[2] > 0.60 and ext[3] < 0.40 and ext[4] < 0.40:
                score = 0.93
        elif target == "W":
            if ext[1] > 0.55 and ext[2] > 0.55 and ext[3] > 0.55 and ext[4] < 0.40:
                score = 0.93

        return float(np.clip(score, 0.15, 0.98))

    def draw_hud(self, frame, score):
        h, w, _ = frame.shape
        target = TARGET_KEYS[self.target_idx]

        gauge_color = (0, 0, 255) if score < 0.6 else ((0, 255, 255) if score < 0.85 else (0, 255, 0))
        bar_width = int((w - 80) * score)

        cv2.rectangle(frame, (40, 20), (w - 40, 45), (40, 40, 40), -1)
        cv2.rectangle(frame, (40, 20), (40 + bar_width, 45), gauge_color, -1)
        cv2.putText(frame, f"Match Score: {int(score * 100)}%", (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gauge_color, 2)

        if score >= 0.85:
            cv2.rectangle(frame, (0, h - 50), (w, h), (0, 180, 0), -1)
            cv2.putText(frame, f"GESTURE MATCHED: '{target}'", (int(w * 0.20), h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self._init_detector()

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        self.frame_counter += 1
        score = 0.0

        if self.frame_counter % 2 == 0:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            detection_result = self.detector.detect(mp_image)

            if detection_result.hand_landmarks:
                landmarks = detection_result.hand_landmarks[0]
                self.draw_skeleton(img, landmarks)
                score = self.evaluate_gesture(landmarks)

            self.score_buffer.append(score)

        smoothed_score = float(np.mean(self.score_buffer)) if self.score_buffer else 0.0
        self.draw_hud(img, smoothed_score)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------------------------------------------------------
# Streamlit Interface
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Sign Language Trainer", layout="wide")

st.title("Sign Language Recognition Trainer")

if "target_idx" not in st.session_state:
    st.session_state.target_idx = 0

col1, col2 = st.columns([2.5, 1])

with col2:
    st.subheader("Target Gesture Selection")
    current_target = TARGET_KEYS[st.session_state.target_idx]
    st.metric(label="Current Target Sign", value=f"Letter '{current_target}'")
    st.info(f"**Instructions:** {TARGET_SIGNS[current_target]}")

    c1, c2 = st.columns(2)
    if c1.button("Previous Sign"):
        st.session_state.target_idx = (st.session_state.target_idx - 1) % len(TARGET_KEYS)
        st.rerun()

    if c2.button("Next Sign"):
        st.session_state.target_idx = (st.session_state.target_idx + 1) % len(TARGET_KEYS)
        st.rerun()

with col1:
    webrtc_ctx = webrtc_streamer(
        key="sign-trainer",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=SignTrainerProcessor,
        media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
        async_processing=False,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.set_target_idx(st.session_state.target_idx)
