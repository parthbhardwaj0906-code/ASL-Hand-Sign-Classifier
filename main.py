import os
import sys
import time
from collections import deque
import cv2
import numpy as np

# MediaPipe Python Tasks Vision (used for native desktop mode)
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    mp = None
    python = None
    vision = None

# Streamlit (used for web deployment)
try:
    import streamlit as st
    import streamlit.components.v1 as components
except ImportError:
    st = None
    components = None

# =============================================================================
# EXACT ASL SIGN LANGUAGE CONFIGURATION & LETTER MAPPINGS
# Target gestures: A, B, C, D, I, L, O, V, W, Y
# =============================================================================
TARGET_SIGNS = {
    "A": "Fist with thumb resting along the side of the index finger",
    "B": "Four fingers straight up and flat together, thumb tucked across palm",
    "C": "All fingers curved forming a wide C-shape",
    "D": "Index finger straight up, thumb touching tips of middle, ring, and pinky fingers",
    "I": "Pinky extended straight up, all other fingers folded in, thumb tucked",
    "L": "Index and thumb extended outward at 90 degrees",
    "O": "All fingertips curved touching thumb tip to form a circle",
    "V": "Index and middle fingers extended apart in a V-shape",
    "W": "Index, middle, and ring fingers extended upward",
    "Y": "Thumb and pinky extended wide outward, middle three fingers folded",
}
TARGET_KEYS = list(TARGET_SIGNS.keys())

# Standard MediaPipe Hand Connections for bone visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm
]


# =============================================================================
# DESKTOP NATIVE OPENCV ENGINE (Used when running: python main.py)
# =============================================================================
class AsyncHandLandmarker:
    def __init__(self, model_path="hand_landmarker.task"):
        self.latest_landmarks = None
        if not os.path.exists(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(script_dir, "hand_landmarker.task")
            if os.path.exists(alt_path):
                model_path = alt_path

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            # Lowered thresholds to keep tracking during fast movements/motion blur
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35,
            result_callback=self._result_callback,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _result_callback(self, result, output_image, timestamp_ms):
        if result and result.hand_landmarks:
            self.latest_landmarks = result.hand_landmarks[0]
        else:
            self.latest_landmarks = None

    def process_frame_async(self, frame_rgb, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self.detector.detect_async(mp_image, timestamp_ms)

    def get_landmarks(self):
        return self.latest_landmarks

    def close(self):
        self.detector.close()


class PresentationSignTrainer:
    def __init__(self, model_path="hand_landmarker.task"):
        self.detector = AsyncHandLandmarker(model_path=model_path)

        self.current_idx = 0
        self.score_buffer = deque(maxlen=5)
        self.force_success = False
        self.success_timer = 0

        self.btn_coords = [0, 0, 0, 0]
        self.btn_clicked = False

        # Persistence memory to prevent flickering/turning off on brief dropped frames
        self.last_valid_landmarks = None
        self.landmark_hold_counter = 0

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1, x2, y2 = self.btn_coords
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.btn_clicked = True

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
        target = TARGET_KEYS[self.current_idx]
        ext = self.calculate_finger_extensions(landmarks)
        thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext = ext

        # Strict 2D normalized coordinates to prevent depth-scaling corruption
        wrist = np.array([landmarks[0].x, landmarks[0].y])
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
        index_mcp = np.array([landmarks[5].x, landmarks[5].y])

        middle_pip = np.array([landmarks[10].x, landmarks[10].y])
        ring_pip = np.array([landmarks[14].x, landmarks[14].y])

        index_tip = np.array([landmarks[8].x, landmarks[8].y])
        middle_tip = np.array([landmarks[12].x, landmarks[12].y])
        ring_tip = np.array([landmarks[16].x, landmarks[16].y])
        pinky_tip = np.array([landmarks[20].x, landmarks[20].y])

        palm_size = np.linalg.norm(index_mcp - wrist)
        if palm_size == 0:
            palm_size = 0.20

        norm_thumb_to_mcp = np.linalg.norm(thumb_tip - index_mcp) / palm_size
        norm_thumb_to_pinky = np.linalg.norm(thumb_tip - pinky_tip) / palm_size

        score = 0.20

        if target == "A":
            four_fingers_folded = all(e < 0.35 for e in ext[1:])
            thumb_tucked_side = norm_thumb_to_mcp < 0.65
            if four_fingers_folded and thumb_tucked_side and pinky_ext < 0.35:
                score = 0.95
            elif four_fingers_folded:
                score = 0.40

        elif target == "B":
            four_upright = all(e > 0.60 for e in ext[1:])
            thumb_tucked = (norm_thumb_to_pinky < 1.10 or norm_thumb_to_mcp < 0.65) and thumb_ext < 0.50
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

            fingers_curved = all(0.8 < d < 2.3 for d in [d_index, d_middle, d_ring, d_pinky])
            thumb_index_gap = np.linalg.norm(thumb_tip - index_tip) / palm_size
            thumb_middle_gap = np.linalg.norm(thumb_tip - middle_tip) / palm_size
            is_open_arc = 0.35 < thumb_index_gap < 1.95 and 0.35 < thumb_middle_gap < 1.95

            if fingers_curved and is_open_arc:
                score = 0.94
            elif fingers_curved:
                score = 0.50

        elif target == "D":
            index_pointing_up = index_ext > 0.55
            other_three_folded = middle_ext < 0.45 and ring_ext < 0.45 and pinky_ext < 0.45
            avg_dist_to_thumb = np.mean([
                np.linalg.norm(thumb_tip - middle_tip) / palm_size,
                np.linalg.norm(thumb_tip - ring_tip) / palm_size,
            ])
            thumb_loop = avg_dist_to_thumb < 0.65

            if index_pointing_up and other_three_folded and thumb_loop:
                score = 0.95
            elif index_pointing_up and other_three_folded:
                score = 0.50

        elif target == "I":
            pinky_extended = pinky_ext > 0.50
            other_three_folded = index_ext < 0.45 and middle_ext < 0.45 and ring_ext < 0.45
            dist_to_middle_pip = np.linalg.norm(thumb_tip - middle_pip) / palm_size
            dist_to_ring_pip = np.linalg.norm(thumb_tip - ring_pip) / palm_size
            thumb_tucked = (dist_to_middle_pip < 0.85 or dist_to_ring_pip < 0.85 or norm_thumb_to_mcp < 0.80) and norm_thumb_to_pinky < 1.20

            if pinky_extended and other_three_folded and thumb_tucked:
                score = 0.95
            elif pinky_extended and other_three_folded:
                score = 0.35

        elif target == "L":
            if index_ext > 0.50 and all(e < 0.45 for e in ext[2:]) and norm_thumb_to_mcp > 0.70:
                score = 0.94

        elif target == "O":
            tips = [index_tip, middle_tip, ring_tip, pinky_tip]
            avg_dist_to_thumb = np.mean([np.linalg.norm(thumb_tip - t) / palm_size for t in tips])
            if avg_dist_to_thumb < 0.55:
                score = 0.93

        elif target == "V":
            index_middle_up = index_ext > 0.55 and middle_ext > 0.55
            ring_pinky_folded = ring_ext < 0.45 and pinky_ext < 0.45
            thumb_tucked = norm_thumb_to_mcp < 0.75 and thumb_ext < 0.50

            fingers_gap = np.linalg.norm(index_tip - middle_tip) / palm_size
            fingers_apart = fingers_gap >= 0.35

            if index_middle_up and ring_pinky_folded and thumb_tucked and fingers_apart:
                score = 0.93
            elif index_middle_up and ring_pinky_folded:
                score = 0.35

        elif target == "W":
            three_upright = index_ext > 0.50 and middle_ext > 0.50 and ring_ext > 0.50
            pinky_folded = pinky_ext < 0.45
            thumb_tucked = norm_thumb_to_mcp < 0.75 and thumb_ext < 0.50

            if three_upright and pinky_folded and thumb_tucked:
                score = 0.93
            elif three_upright and pinky_folded:
                score = 0.35

        elif target == "Y":
            pinky_extended = pinky_ext > 0.50
            middle_three_folded = index_ext < 0.45 and middle_ext < 0.45 and ring_ext < 0.45
            dist_to_middle_pip = np.linalg.norm(thumb_tip - middle_pip) / palm_size
            thumb_strictly_extended = (norm_thumb_to_mcp > 0.65) and (dist_to_middle_pip > 0.60) and (norm_thumb_to_pinky > 1.0)

            if pinky_extended and middle_three_folded and thumb_strictly_extended:
                score = 0.95
            elif pinky_extended and middle_three_folded:
                score = 0.35

        return float(np.clip(score, 0.15, 0.98))

    def draw_skeleton(self, frame, landmarks):
        h, w, _ = frame.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(frame, points[p1], points[p2], (255, 255, 255), 2)

        for pt in points:
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)

    def draw_hud(self, frame, score, hand_detected):
        h, w, _ = frame.shape
        target = TARGET_KEYS[self.current_idx]
        desc = TARGET_SIGNS[target]

        cv2.rectangle(frame, (20, 20), (w - 200, 100), (30, 30, 30), -1)
        cv2.rectangle(frame, (20, 20), (w - 200, 100), (200, 200, 200), 1)
        cv2.putText(frame, f"TARGET SIGN: {target}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.putText(frame, desc, (40, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        btn_x1, btn_y1, btn_x2, btn_y2 = w - 180, 20, w - 20, 100
        self.btn_coords = [btn_x1, btn_y1, btn_x2, btn_y2]
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 120, 255), -1)
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 255, 255), 2)
        cv2.putText(frame, "NEXT SIGN >", (btn_x1 + 12, btn_y1 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        gauge_color = (0, 0, 255) if score < 0.6 else ((0, 255, 255) if score < 0.85 else (0, 255, 0))
        bar_width = int((w - 80) * score)
        cv2.rectangle(frame, (40, 120), (w - 40, 140), (50, 50, 50), -1)
        cv2.rectangle(frame, (40, 120), (40 + bar_width, 140), gauge_color, -1)
        cv2.putText(frame, f"Match Confidence: {int(score * 100)}%", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gauge_color, 2)

        if score >= 0.85 or self.success_timer > 0:
            cv2.rectangle(frame, (0, h - 80), (w, h), (0, 180, 0), -1)
            cv2.putText(frame, f"GESTURE MATCHED: '{target}' DETECTED!", (w // 6, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            if self.success_timer > 0:
                self.success_timer -= 1

        status = "Hand Landmarks Active" if hand_detected else "Show Hand to Camera"
        cv2.putText(frame, status, (w - 280, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if hand_detected else (0, 0, 255), 1)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        window_name = "Sign Language Trainer HUD"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int((time.time() - start_time) * 1000)
            self.detector.process_frame_async(rgb_frame, timestamp_ms)

            landmarks = self.detector.get_landmarks()

            # Decay buffer: Hold landmarks for up to 4 frames if lost briefly due to blur
            if landmarks is not None:
                self.last_valid_landmarks = landmarks
                self.landmark_hold_counter = 4
            elif self.landmark_hold_counter > 0:
                landmarks = self.last_valid_landmarks
                self.landmark_hold_counter -= 1

            score = 0.0
            hand_detected = False

            if landmarks:
                hand_detected = True
                self.draw_skeleton(frame, landmarks)
                score = self.evaluate_gesture(landmarks)

            if self.btn_clicked:
                self.current_idx = (self.current_idx + 1) % len(TARGET_KEYS)
                self.score_buffer.clear()
                self.btn_clicked = False

            if self.force_success:
                score = 1.0
                self.success_timer = 25
                self.force_success = False

            self.score_buffer.append(score)
            smoothed_score = float(np.mean(self.score_buffer)) if self.score_buffer else 0.0

            self.draw_hud(frame, smoothed_score, hand_detected)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                self.force_success = True
            elif key == ord("n"):
                self.btn_clicked = True
            elif key == ord("p"):
                self.current_idx = (self.current_idx - 1) % len(TARGET_KEYS)
                self.score_buffer.clear()

        self.detector.close()
        cap.release()
        cv2.destroyAllWindows()


# =============================================================================
# STREAMLIT WEB APPLICATION (Used for Cloud Deployment & Browser Access)
# =============================================================================
def build_html_trainer(current_idx: int) -> str:
    """
    Renders Google's official MediaPipe Tasks Vision (HandLandmarker) with exact
    1:1 mathematical fidelity to the Python evaluation rules for letters A, B, C, D, I, L, O, V, W, Y.
    Runs client-side via WebAssembly/WebGL for 60 FPS performance without WebRTC connection issues.
    """
    target = TARGET_KEYS[current_idx]
    desc = TARGET_SIGNS[target]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ASL Trainer</title>
  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      background: #0f1117;
      color: #fafafa;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 6px;
      overflow: hidden;
    }}
    .stage {{
      position: relative;
      width: 640px;
      height: 480px;
      background: #1a1c24;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5);
      border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    video#webcam {{
      display: none;
    }}
    canvas#viewport {{
      position: absolute;
      top: 0;
      left: 0;
      width: 640px;
      height: 480px;
      display: block;
    }}
    .loader-overlay {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(15, 17, 23, 0.92);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      z-index: 50;
      transition: opacity 0.3s ease;
    }}
    .spinner {{
      width: 48px;
      height: 48px;
      border: 4px solid rgba(255, 255, 255, 0.15);
      border-top-color: #ff7800;
      border-radius: 50%;
      animation: spin 1s linear infinite;
      margin-bottom: 16px;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
    .loader-text {{
      font-size: 15px;
      font-weight: 500;
      color: #e0e0e0;
      text-align: center;
      max-width: 80%;
      line-height: 1.4;
    }}
  </style>
</head>
<body>
  <div class="stage">
    <video id="webcam" autoplay playsinline muted></video>
    <canvas id="viewport" width="640" height="480"></canvas>
    <div id="loader" class="loader-overlay">
      <div class="spinner"></div>
      <div id="loader-text" class="loader-text">Loading MediaPipe HandLandmarker Model...</div>
    </div>
  </div>

  <!-- MediaPipe Tasks Vision Official Bundle -->
  <script type="module">
    import {{
      FilesetResolver,
      HandLandmarker
    }} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

    const TARGET_SIGNS = {repr(TARGET_SIGNS)};
    const TARGET_KEYS = {repr(TARGET_KEYS)};
    const HAND_CONNECTIONS = {repr(HAND_CONNECTIONS)};

    let currentIdx = {current_idx};
    let successTimer = 0;
    let forceSuccess = false;

    // 4-frame decay buffer to hold landmarks briefly on motion blur
    let lastValidLandmarks = null;
    let landmarkHoldCounter = 0;

    // 5-frame moving average score buffer
    const scoreBuffer = [];

    const videoElement = document.getElementById("webcam");
    const canvasElement = document.getElementById("viewport");
    const canvasCtx = canvasElement.getContext("2d");
    const loader = document.getElementById("loader");
    const loaderText = document.getElementById("loader-text");

    let handLandmarker = null;
    let lastVideoTime = -1;

    // Euclidean distance helper (identical to np.linalg.norm on 2D coordinates)
    function dist2D(p1, p2) {{
      const dx = p1.x - p2.x;
      const dy = p1.y - p2.y;
      return Math.hypot(dx, dy);
    }}

    function clamp(val, min, max) {{
      return Math.max(min, Math.min(max, val));
    }}

    // 1:1 mathematical port of PresentationSignTrainer.calculate_finger_extensions
    function calculateFingerExtensions(landmarks) {{
      const wrist = landmarks[0];
      const fingerIndices = [[4, 2], [8, 6], [12, 10], [16, 14], [20, 18]];

      const extensions = [];
      const thumbTip = landmarks[4];
      const thumbDist = dist2D(thumbTip, wrist);
      extensions.push(clamp((thumbDist - 0.15) / 0.18, 0.0, 1.0));

      for (let i = 1; i < fingerIndices.length; i++) {{
        const [tipIdx, pipIdx] = fingerIndices[i];
        const tip = landmarks[tipIdx];
        const pip = landmarks[pipIdx];
        const diff = pip.y - tip.y;
        extensions.push(clamp((diff + 0.02) / 0.12, 0.0, 1.0));
      }}

      return extensions;
    }}

    // 1:1 mathematical port of PresentationSignTrainer.evaluate_gesture
    function evaluateGesture(landmarks) {{
      const target = TARGET_KEYS[currentIdx];
      const ext = calculateFingerExtensions(landmarks);
      const [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext] = ext;

      // Strict 2D normalized coordinates to prevent depth-scaling corruption
      const wrist = landmarks[0];
      const thumb_tip = landmarks[4];
      const index_mcp = landmarks[5];

      const middle_pip = landmarks[10];
      const ring_pip = landmarks[14];

      const index_tip = landmarks[8];
      const middle_tip = landmarks[12];
      const ring_tip = landmarks[16];
      const pinky_tip = landmarks[20];

      let palm_size = dist2D(index_mcp, wrist);
      if (palm_size === 0) {{
        palm_size = 0.20;
      }}

      const norm_thumb_to_mcp = dist2D(thumb_tip, index_mcp) / palm_size;
      const norm_thumb_to_pinky = dist2D(thumb_tip, pinky_tip) / palm_size;

      let score = 0.20;

      if (target === "A") {{
        const four_fingers_folded = ext.slice(1).every(e => e < 0.35);
        const thumb_tucked_side = norm_thumb_to_mcp < 0.65;
        if (four_fingers_folded && thumb_tucked_side && pinky_ext < 0.35) {{
          score = 0.95;
        }} else if (four_fingers_folded) {{
          score = 0.40;
        }}
      }} else if (target === "B") {{
        const four_upright = ext.slice(1).every(e => e > 0.60);
        const thumb_tucked = (norm_thumb_to_pinky < 1.10 || norm_thumb_to_mcp < 0.65) && thumb_ext < 0.50;
        const is_open_hand = ext.every(e => e > 0.60);
        if (four_upright && thumb_tucked && !is_open_hand) {{
          score = 0.94;
        }} else if (four_upright && !is_open_hand) {{
          score = 0.45;
        }}
      }} else if (target === "C") {{
        const d_index = dist2D(index_tip, wrist) / palm_size;
        const d_middle = dist2D(middle_tip, wrist) / palm_size;
        const d_ring = dist2D(ring_tip, wrist) / palm_size;
        const d_pinky = dist2D(pinky_tip, wrist) / palm_size;

        const fingers_curved = [d_index, d_middle, d_ring, d_pinky].every(d => d > 0.8 && d < 2.3);
        const thumb_index_gap = dist2D(thumb_tip, index_tip) / palm_size;
        const thumb_middle_gap = dist2D(thumb_tip, middle_tip) / palm_size;
        const is_open_arc = (thumb_index_gap > 0.35 && thumb_index_gap < 1.95) && (thumb_middle_gap > 0.35 && thumb_middle_gap < 1.95);

        if (fingers_curved && is_open_arc) {{
          score = 0.94;
        }} else if (fingers_curved) {{
          score = 0.50;
        }}
      }} else if (target === "D") {{
        const index_pointing_up = index_ext > 0.55;
        const other_three_folded = middle_ext < 0.45 && ring_ext < 0.45 && pinky_ext < 0.45;
        const avg_dist_to_thumb = (
          (dist2D(thumb_tip, middle_tip) / palm_size) +
          (dist2D(thumb_tip, ring_tip) / palm_size)
        ) / 2.0;
        const thumb_loop = avg_dist_to_thumb < 0.65;

        if (index_pointing_up && other_three_folded && thumb_loop) {{
          score = 0.95;
        }} else if (index_pointing_up && other_three_folded) {{
          score = 0.50;
        }}
      }} else if (target === "I") {{
        const pinky_extended = pinky_ext > 0.50;
        const other_three_folded = index_ext < 0.45 && middle_ext < 0.45 && ring_ext < 0.45;
        const dist_to_middle_pip = dist2D(thumb_tip, middle_pip) / palm_size;
        const dist_to_ring_pip = dist2D(thumb_tip, ring_pip) / palm_size;
        const thumb_tucked = (dist_to_middle_pip < 0.85 || dist_to_ring_pip < 0.85 || norm_thumb_to_mcp < 0.80) && norm_thumb_to_pinky < 1.20;

        if (pinky_extended && other_three_folded && thumb_tucked) {{
          score = 0.95;
        }} else if (pinky_extended && other_three_folded) {{
          score = 0.35;
        }}
      }} else if (target === "L") {{
        if (index_ext > 0.50 && ext.slice(2).every(e => e < 0.45) && norm_thumb_to_mcp > 0.70) {{
          score = 0.94;
        }}
      }} else if (target === "O") {{
        const tips = [index_tip, middle_tip, ring_tip, pinky_tip];
        const avg_dist_to_thumb = tips.reduce((acc, t) => acc + (dist2D(thumb_tip, t) / palm_size), 0) / tips.length;
        if (avg_dist_to_thumb < 0.55) {{
          score = 0.93;
        }}
      }} else if (target === "V") {{
        const index_middle_up = index_ext > 0.55 && middle_ext > 0.55;
        const ring_pinky_folded = ring_ext < 0.45 && pinky_ext < 0.45;
        const thumb_tucked = norm_thumb_to_mcp < 0.75 && thumb_ext < 0.50;

        const fingers_gap = dist2D(index_tip, middle_tip) / palm_size;
        const fingers_apart = fingers_gap >= 0.35;

        if (index_middle_up && ring_pinky_folded && thumb_tucked && fingers_apart) {{
          score = 0.93;
        }} else if (index_middle_up && ring_pinky_folded) {{
          score = 0.35;
        }}
      }} else if (target === "W") {{
        const three_upright = index_ext > 0.50 && middle_ext > 0.50 && ring_ext > 0.50;
        const pinky_folded = pinky_ext < 0.45;
        const thumb_tucked = norm_thumb_to_mcp < 0.75 && thumb_ext < 0.50;

        if (three_upright && pinky_folded && thumb_tucked) {{
          score = 0.93;
        }} else if (three_upright && pinky_folded) {{
          score = 0.35;
        }}
      }} else if (target === "Y") {{
        const pinky_extended = pinky_ext > 0.50;
        const middle_three_folded = index_ext < 0.45 && middle_ext < 0.45 && ring_ext < 0.45;
        const dist_to_middle_pip = dist2D(thumb_tip, middle_pip) / palm_size;
        const thumb_strictly_extended = (norm_thumb_to_mcp > 0.65) && (dist_to_middle_pip > 0.60) && (norm_thumb_to_pinky > 1.0);

        if (pinky_extended && middle_three_folded && thumb_strictly_extended) {{
          score = 0.95;
        }} else if (pinky_extended && middle_three_folded) {{
          score = 0.35;
        }}
      }}

      return clamp(score, 0.15, 0.98);
    }}

    // Draw skeleton bone connections and landmark points
    function drawSkeleton(landmarks, w, h) {{
      canvasCtx.lineWidth = 2;
      canvasCtx.strokeStyle = "#FFFFFF";

      for (const [p1, p2] of HAND_CONNECTIONS) {{
        const pt1 = landmarks[p1];
        const pt2 = landmarks[p2];
        canvasCtx.beginPath();
        canvasCtx.moveTo(pt1.x * w, pt1.y * h);
        canvasCtx.lineTo(pt2.x * w, pt2.y * h);
        canvasCtx.stroke();
      }}

      canvasCtx.fillStyle = "#00FF00";
      for (const pt of landmarks) {{
        canvasCtx.beginPath();
        canvasCtx.arc(pt.x * w, pt.y * h, 4, 0, 2 * Math.PI);
        canvasCtx.fill();
      }}
    }}

    // Draw HUD overlays identical to PresentationSignTrainer.draw_hud
    function drawHUD(score, handDetected, w, h) {{
      const target = TARGET_KEYS[currentIdx];
      const desc = TARGET_SIGNS[target];

      // Target Sign Box
      canvasCtx.fillStyle = "rgba(30, 30, 30, 0.9)";
      canvasCtx.fillRect(20, 20, w - 220, 80);
      canvasCtx.strokeStyle = "rgba(200, 200, 200, 0.8)";
      canvasCtx.lineWidth = 1;
      canvasCtx.strokeRect(20, 20, w - 220, 80);

      canvasCtx.font = "bold 26px sans-serif";
      canvasCtx.fillStyle = "#FFFFFF";
      canvasCtx.fillText(`TARGET SIGN: ${{target}}`, 40, 58);

      canvasCtx.font = "12px sans-serif";
      canvasCtx.fillStyle = "#B4B4B4";
      canvasCtx.fillText(desc, 40, 86);

      // NEXT SIGN > Button
      const btnX1 = w - 180, btnY1 = 20, btnW = 160, btnH = 80;
      canvasCtx.fillStyle = "#0078FF";
      canvasCtx.fillRect(btnX1, btnY1, btnW, btnH);
      canvasCtx.strokeStyle = "#FFFFFF";
      canvasCtx.lineWidth = 2;
      canvasCtx.strokeRect(btnX1, btnY1, btnW, btnH);

      canvasCtx.font = "bold 17px sans-serif";
      canvasCtx.fillStyle = "#FFFFFF";
      canvasCtx.fillText("NEXT SIGN >", btnX1 + 22, btnY1 + 48);

      // Match Confidence Bar
      const gaugeColor = score < 0.6 ? "#FF0000" : (score < 0.85 ? "#FFFF00" : "#00FF00");
      const barWidth = Math.max(0, Math.min(w - 80, (w - 80) * score));

      canvasCtx.fillStyle = "rgba(50, 50, 50, 0.85)";
      canvasCtx.fillRect(40, 120, w - 80, 20);

      canvasCtx.fillStyle = gaugeColor;
      canvasCtx.fillRect(40, 120, barWidth, 20);

      canvasCtx.font = "bold 16px sans-serif";
      canvasCtx.fillStyle = gaugeColor;
      canvasCtx.fillText(`Match Confidence: ${{Math.round(score * 100)}}%`, 40, 165);

      // Success Banner
      if (score >= 0.85 || successTimer > 0) {{
        canvasCtx.fillStyle = "#00B400";
        canvasCtx.fillRect(0, h - 80, w, 80);

        canvasCtx.font = "bold 24px sans-serif";
        canvasCtx.fillStyle = "#FFFFFF";
        canvasCtx.fillText(`GESTURE MATCHED: '${{target}}' DETECTED!`, Math.floor(w / 7), h - 30);

        if (successTimer > 0) {{
          successTimer -= 1;
        }}
      }}

      // Status text
      const statusText = handDetected ? "Hand Landmarks Active" : "Show Hand to Camera";
      canvasCtx.font = "14px sans-serif";
      canvasCtx.fillStyle = handDetected ? "#00FF00" : "#FF0000";
      canvasCtx.fillText(statusText, w - 240, h - 90);
    }}

    // Mouse click handling for NEXT SIGN > on canvas
    canvasElement.addEventListener("click", (e) => {{
      const rect = canvasElement.getBoundingClientRect();
      const scaleX = canvasElement.width / rect.width;
      const scaleY = canvasElement.height / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      const btnX1 = canvasElement.width - 180;
      const btnX2 = canvasElement.width - 20;
      const btnY1 = 20;
      const btnY2 = 100;

      if (x >= btnX1 && x <= btnX2 && y >= btnY1 && y <= btnY2) {{
        currentIdx = (currentIdx + 1) % TARGET_KEYS.length;
        scoreBuffer.length = 0;
      }}
    }});

    // Keyboard navigation ('n' = next, 'p' = previous, space = test match)
    window.addEventListener("keydown", (e) => {{
      if (e.key === "n" || e.key === "N") {{
        currentIdx = (currentIdx + 1) % TARGET_KEYS.length;
        scoreBuffer.length = 0;
      }} else if (e.key === "p" || e.key === "P") {{
        currentIdx = (currentIdx - 1 + TARGET_KEYS.length) % TARGET_KEYS.length;
        scoreBuffer.length = 0;
      }} else if (e.key === " ") {{
        forceSuccess = true;
      }}
    }});

    async function initTrainer() {{
      try {{
        loaderText.innerText = "Initializing MediaPipe Vision Tasks...";
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );

        loaderText.innerText = "Loading HandLandmarker model (hand_landmarker.task)...";
        const modelUrl = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

        try {{
          handLandmarker = await HandLandmarker.createFromOptions(vision, {{
            baseOptions: {{
              modelAssetPath: modelUrl,
              delegate: "GPU"
            }},
            runningMode: "VIDEO",
            numHands: 1,
            minHandDetectionConfidence: 0.35,
            minHandPresenceConfidence: 0.35,
            minTrackingConfidence: 0.35
          }});
        }} catch (gpuErr) {{
          console.warn("GPU delegate unavailable, falling back to CPU", gpuErr);
          handLandmarker = await HandLandmarker.createFromOptions(vision, {{
            baseOptions: {{
              modelAssetPath: modelUrl,
              delegate: "CPU"
            }},
            runningMode: "VIDEO",
            numHands: 1,
            minHandDetectionConfidence: 0.35,
            minHandPresenceConfidence: 0.35,
            minTrackingConfidence: 0.35
          }});
        }}

        loaderText.innerText = "Requesting Webcam access...";
        const stream = await navigator.mediaDevices.getUserMedia({{
          video: {{
            width: {{ ideal: 640 }},
            height: {{ ideal: 480 }},
            frameRate: {{ ideal: 30 }}
          }},
          audio: false
        }});

        videoElement.srcObject = stream;
        videoElement.onloadedmetadata = () => {{
          videoElement.play();
          loader.style.opacity = "0";
          setTimeout(() => {{ loader.style.display = "none"; }}, 300);
          requestAnimationFrame(renderFrame);
        }};
      }} catch (err) {{
        console.error("Initialization error:", err);
        loaderText.innerText = "Webcam or Model initialization error: " + err.message + "\\nPlease allow camera permissions.";
      }}
    }}

    function renderFrame() {{
      const w = canvasElement.width;
      const h = canvasElement.height;

      if (videoElement.readyState >= 2) {{
        // Draw video mirrored (flip horizontally, identical to cv2.flip(frame, 1))
        canvasCtx.save();
        canvasCtx.translate(w, 0);
        canvasCtx.scale(-1, 1);
        canvasCtx.drawImage(videoElement, 0, 0, w, h);
        canvasCtx.restore();

        let landmarks = null;
        if (handLandmarker && videoElement.currentTime !== lastVideoTime) {{
          lastVideoTime = videoElement.currentTime;
          const startTimeMs = performance.now();
          const results = handLandmarker.detectForVideo(canvasElement, startTimeMs);

          if (results && results.landmarks && results.landmarks.length > 0) {{
            landmarks = results.landmarks[0];
          }}
        }}

        // Decay buffer: Hold landmarks for up to 4 frames if lost briefly due to blur
        if (landmarks !== null) {{
          lastValidLandmarks = landmarks;
          landmarkHoldCounter = 4;
        }} else if (landmarkHoldCounter > 0) {{
          landmarks = lastValidLandmarks;
          landmarkHoldCounter -= 1;
        }}

        let score = 0.0;
        let handDetected = false;

        if (landmarks) {{
          handDetected = true;
          drawSkeleton(landmarks, w, h);
          score = evaluateGesture(landmarks);
        }}

        if (forceSuccess) {{
          score = 1.0;
          successTimer = 25;
          forceSuccess = false;
        }}

        scoreBuffer.push(score);
        if (scoreBuffer.length > 5) {{
          scoreBuffer.shift();
        }}
        const smoothedScore = scoreBuffer.length > 0
          ? scoreBuffer.reduce((a, b) => a + b, 0) / scoreBuffer.length
          : 0.0;

        drawHUD(smoothedScore, handDetected, w, h);
      }}

      requestAnimationFrame(renderFrame);
    }}

    initTrainer();
  </script>
</body>
</html>"""


def run_streamlit_app():
    st.set_page_config(
        page_title="ASL Sign Language Recognition Trainer",
        page_icon="🤟",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .metric-card {
            background-color: #1E222D;
            border: 1px solid #313642;
            padding: 1rem 1.25rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🤟 ASL Hand Sign Language Recognition Trainer")
    st.markdown(
        "Real-time American Sign Language recognition powered by **MediaPipe Tasks Vision** (`hand_landmarker.task`)."
    )

    if "target_idx" not in st.session_state:
        st.session_state.target_idx = 0

    col_cam, col_ctrl = st.columns([2.6, 1.4])

    with col_ctrl:
        st.subheader("🎯 Sign Practice Control")

        selected_key = st.selectbox(
            "Select Sign to Practice",
            TARGET_KEYS,
            index=st.session_state.target_idx,
            key="sign_selector",
        )
        st.session_state.target_idx = TARGET_KEYS.index(selected_key)

        curr_target = TARGET_KEYS[st.session_state.target_idx]
        curr_desc = TARGET_SIGNS[curr_target]

        st.info(f"**Target Gesture: Letter '{curr_target}'**\n\n{curr_desc}")

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("◀ Previous Sign", use_container_width=True):
                st.session_state.target_idx = (st.session_state.target_idx - 1) % len(TARGET_KEYS)
                st.rerun()
        with btn_c2:
            if st.button("Next Sign ▶", use_container_width=True):
                st.session_state.target_idx = (st.session_state.target_idx + 1) % len(TARGET_KEYS)
                st.rerun()

        st.divider()
        st.markdown(
            """
            ### 💡 Tips & Hotkeys
            - **Click `NEXT SIGN >`** on the video HUD or use buttons to cycle letters.
            - **Keys**: Press `N` for next sign, `P` for previous sign.
            - Hold hand ~1.5 to 2.5 feet from the camera for best accuracy.
            - Supported letters: **A, B, C, D, I, L, O, V, W, Y**.
            """
        )

    with col_cam:
        html_code = build_html_trainer(st.session_state.target_idx)
        components.html(html_code, height=510)


# =============================================================================
# ENTRYPOINT
# =============================================================================
def is_streamlit_running() -> bool:
    """Detects if execution context is active inside Streamlit."""
    if st is None:
        return False
    try:
        return hasattr(st, "runtime") and st.runtime.exists()
    except Exception:
        return False


if __name__ == "__main__":
    if is_streamlit_running():
        run_streamlit_app()
    elif "--streamlit" in sys.argv or (len(sys.argv) > 1 and sys.argv[1] == "run"):
        # Explicit fallback if invoked via streamlit
        run_streamlit_app()
    else:
        # Direct Python CLI launch: Run the native desktop OpenCV HUD
        try:
            app = PresentationSignTrainer()
            app.run()
        except Exception as e:
            # If camera or display is unavailable, provide instructions
            print(f"[Notice] Desktop OpenCV window exited: {e}")
            print("\nTo run the interactive web application, execute:")
            print("    streamlit run main.py\n")
elif is_streamlit_running():
    # When imported or executed by Streamlit runner
    run_streamlit_app()