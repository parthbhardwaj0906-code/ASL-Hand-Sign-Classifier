import os
import queue
import time
from collections import deque

import av
import cv2
import numpy as np
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
# Absolute Path Resolution for Deployment Stability
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

# -----------------------------------------------------------------------------
# WebRTC & Media Stream Configuration
# -----------------------------------------------------------------------------
MEDIA_STREAM_CONSTRAINTS = {
    "video": {
        "width": {"ideal": 640, "max": 854},
        "height": {"ideal": 480, "max": 480},
        "frameRate": {"ideal": 30, "min": 20},
    },
    "audio": False,
}

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -----------------------------------------------------------------------------
# Sign Language Configuration & Constants
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
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm
]

# -----------------------------------------------------------------------------
# High-Performance WebRTC Video Processor Class
# -----------------------------------------------------------------------------
class SignTrainerProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.detector = None
        self.target_idx = 0
        self.score_buffer = deque(maxlen=5)

    def _init_detector(self):
        """Lazy initializer to prevent thread initialization and dlopen crashes."""
        if self.detector is None:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"MediaPipe task model missing at path: {MODEL_PATH}")
            
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)

    def set_target_idx(self, idx: int):
        self.target_idx = idx
        self.score_buffer.clear()

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
        thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext = ext

        wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
        index_mcp = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
        index_pip = np.array([landmarks[6].x, landmarks[6].y, landmarks[6].z])

        middle_pip = np.array([landmarks[10].x, landmarks[10].y, landmarks[10].z])
        ring_pip = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])

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
            four_fingers_folded = all(e < 0.35 for e in ext[1:])
            thumb_touching_index = (norm_thumb_to_mcp < 0.38) or (norm_thumb_to_pip < 0.38)
            if four_fingers_folded and thumb_touching_index and pinky_ext < 0.30:
                score = 0.95
            elif four_fingers_folded:
                score = 0.40

        elif target == "B":
            four_upright = all(e > 0.65 for e in ext[1:])
            thumb_tucked = (norm_thumb_to_pinky < 0.95 or norm_thumb_to_mcp < 0.50) and thumb_ext < 0.45
            is_open_hand = all(e > 0.60 for e in ext)
            if four_upright and thumb_tucked and not is_open_hand:
                score = 0.94
            elif four_upright and not is_open_hand:
                score = 0.45

        elif target == "C":
            d_index = np.linalg.norm(index_tip - wrist) / palm_size
            d_middle = np.linalg.norm(middle_tip - wrist) / palm_size
            d_ring = np.linalg.norm(ring_tip - wrist) / palm_size
            d_pinky = np.linalg.norm(pinky_tip - wrist) / palm_size

            fingers_curved = all(1.0 < d < 2.1 for d in [d_index, d_middle, d_ring, d_pinky])
            thumb_index_gap = np.linalg.norm(thumb_tip - index_tip) / palm_size
            thumb_middle_gap = np.linalg.norm(thumb_tip - middle_tip) / palm_size
            is_open_arc = 0.48 < thumb_index_gap < 1.75 and 0.48 < thumb_middle_gap < 1.75

            if fingers_curved and is_open_arc:
                score = 0.94
            elif fingers_curved:
                score = 0.50

        elif target == "D":
            index_pointing_up = index_ext > 0.60
            other_three_folded = middle_ext < 0.45 and ring_ext < 0.45 and pinky_ext < 0.45
            avg_dist_to_thumb = np.mean([
                np.linalg.norm(thumb_tip - middle_tip) / palm_size,
                np.linalg.norm(thumb_tip - ring_tip) / palm_size,
            ])
            thumb_loop = avg_dist_to_thumb < 0.55

            if index_pointing_up and other_three_folded and thumb_loop:
                score = 0.95
            elif index_pointing_up and other_three_folded:
                score = 0.50

        elif target == "I":
            pinky_extended = pinky_ext > 0.55
            other_three_folded = index_ext < 0.45 and middle_ext < 0.45 and ring_ext < 0.45
            dist_to_middle_pip = np.linalg.norm(thumb_tip - middle_pip) / palm_size
            dist_to_ring_pip = np.linalg.norm(thumb_tip - ring_pip) / palm_size
            thumb_tucked = (dist_to_middle_pip < 0.75 or dist_to_ring_pip < 0.75 or norm_thumb_to_mcp < 0.72) and norm_thumb_to_pinky < 1.10

            if pinky_extended and other_three_folded and thumb_tucked:
                score = 0.95
            elif pinky_extended and other_three_folded:
                score = 0.35

        elif target == "L":
            if index_ext > 0.55 and all(e < 0.40 for e in ext[2:]) and norm_thumb_to_mcp > 0.85:
                score = 0.94

        elif target == "O":
            tips = [index_tip, middle_tip, ring_tip, pinky_tip]
            avg_dist_to_thumb = np.mean([np.linalg.norm(thumb_tip - t) / palm_size for t in tips])
            if avg_dist_to_thumb < 0.45:
                score = 0.93

        elif target == "U":
            index_middle_up = index_ext > 0.60 and middle_ext > 0.60
            ring_pinky_folded = ring_ext < 0.40 and pinky_ext < 0.40

            fingers_gap = np.linalg.norm(index_tip - middle_tip) / palm_size
            fingers_together = fingers_gap < 0.40

            if index_middle_up and ring_pinky_folded and fingers_together:
                score = 0.95
            elif index_middle_up and ring_pinky_folded:
                score = 0.40

        elif target == "V":
            index_middle_up = index_ext > 0.60 and middle_ext > 0.60
            ring_pinky_folded = ring_ext < 0.40 and pinky_ext < 0.40
            thumb_tucked = norm_thumb_to_mcp < 0.65 and thumb_ext < 0.45

            fingers_gap = np.linalg.norm(index_tip - middle_tip) / palm_size
            fingers_apart = fingers_gap >= 0.40

            if index_middle_up and ring_pinky_folded and thumb_tucked and fingers_apart:
                score = 0.93
            elif index_middle_up and ring_pinky_folded:
                score = 0.35

        elif target == "W":
            three_upright = index_ext > 0.55 and middle_ext > 0.55 and ring_ext > 0.55
            pinky_folded = pinky_ext < 0.40
            thumb_tucked = norm_thumb_to_mcp < 0.65 and thumb_ext < 0.45

            if three_upright and pinky_folded and thumb_tucked:
                score = 0.93
            elif three_upright and pinky_folded:
                score = 0.35

        return float(np.clip(score, 0.15, 0.98))

    def draw_skeleton(self, frame, landmarks):
        h, w, _ = frame.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(frame, points[p1], points[p2], (255, 255, 255), 3)

        for pt in points:
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)

    def draw_hud(self, frame, score, hand_detected):
        h, w, _ = frame.shape
        target = TARGET_KEYS[self.target_idx]

        gauge_color = (0, 0, 255) if score < 0.6 else ((0, 255, 255) if score < 0.85 else (0, 255, 0))
        bar_width = int((w - 80) * score)

        cv2.rectangle(frame, (40, 25), (w - 40, 50), (50, 50, 50), -1)
        cv2.rectangle(frame, (40, 25), (40 + bar_width, 50), gauge_color, -1)
        cv2.putText(frame, f"Match Score: {int(score * 100)}%", (40, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gauge_color, 2)

        if score >= 0.85:
            cv2.rectangle(frame, (0, h - 70), (w, h), (0, 180, 0), -1)
            cv2.putText(frame, f"GESTURE MATCHED: '{target}'", (int(w * 0.20), h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Initialize MediaPipe detector safely on worker thread
        self._init_detector()

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # Downscale for high-speed CPU inference
        h, w, _ = img.shape
        processing_w = 480
        processing_h = int(h * (processing_w / w))
        small_img = cv2.resize(img, (processing_w, processing_h), interpolation=cv2.INTER_LINEAR)

        img_rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        detection_result = self.detector.detect(mp_image)

        score = 0.0
        hand_detected = False

        if detection_result.hand_landmarks:
            hand_detected = True
            landmarks = detection_result.hand_landmarks[0]
            self.draw_skeleton(img, landmarks)
            score = self.evaluate_gesture(landmarks)

        self.score_buffer.append(score)
        smoothed_score = float(np.mean(self.score_buffer)) if self.score_buffer else 0.0

        self.draw_hud(img, smoothed_score, hand_detected)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------------------------------------------------------
# Streamlit Interface
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Sign Language Trainer", layout="wide")

st.title("Sign Language Recognition Trainer")
st.markdown("Real-time webcam pose estimation and gesture evaluation via MediaPipe.")

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
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.set_target_idx(st.session_state.target_idx)
