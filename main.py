import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Sign Language Trainer", layout="wide")

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

if "target_idx" not in st.session_state:
    st.session_state.target_idx = 0

st.title("Sign Language Recognition Trainer")

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
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@0.4/camera_utils.js" crossorigin="anonymous"></script>
      <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.4/drawing_utils.js" crossorigin="anonymous"></script>
      <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/hands.js" crossorigin="anonymous"></script>
      <style>
        .container {{
          position: relative;
          width: 640px;
          height: 480px;
          background: #000;
          border-radius: 8px;
          overflow: hidden;
        }}
        #webcam {{
          position: absolute;
          top: 0;
          left: 0;
          width: 640px;
          height: 480px;
          object-fit: cover;
          transform: scaleX(-1);
        }}
        #canvas {{
          position: absolute;
          top: 0;
          left: 0;
          width: 640px;
          height: 480px;
          z-index: 10;
          transform: scaleX(-1);
        }}
        #hud {{
          position: absolute;
          top: 15px;
          left: 20px;
          right: 20px;
          z-index: 20;
          font-family: sans-serif;
        }}
        .bar-bg {{
          background: rgba(50,50,50,0.8);
          height: 20px;
          border-radius: 10px;
          overflow: hidden;
        }}
        .bar-fill {{
          background: #ff4b4b;
          height: 100%;
          width: 0%;
          transition: width 0.1s;
        }}
        .status {{
          color: #fff;
          margin-top: 8px;
          font-weight: bold;
          font-size: 18px;
          text-shadow: 1px 1px 3px #000;
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <video id="webcam" autoplay playsinline muted></video>
        <canvas id="canvas" width="640" height="480"></canvas>
        <div id="hud">
          <div class="bar-bg"><div id="fill" class="bar-fill"></div></div>
          <div id="status" class="status">Initializing Camera & Hand Model...</div>
        </div>
      </div>

      <script>
        const videoElement = document.getElementById('webcam');
        const canvasElement = document.getElementById('canvas');
        const canvasCtx = canvasElement.getContext('2d');
        const fillBar = document.getElementById('fill');
        const statusText = document.getElementById('status');

        const currentTarget = "{current_target}";

        function evaluateGesture(landmarks) {{
          function ext(tipIdx, pipIdx) {{
            return (landmarks[pipIdx].y - landmarks[tipIdx].y + 0.02) / 0.12;
          }}

          const ext1 = ext(8, 6), ext2 = ext(12, 10), ext3 = ext(16, 14), ext4 = ext(20, 18);
          let score = 0.20;

          if (currentTarget === "A") {{
            if (ext1 < 0.35 && ext2 < 0.35 && ext3 < 0.35 && ext4 < 0.35) score = 0.95;
          }} else if (currentTarget === "B") {{
            if (ext1 > 0.6 && ext2 > 0.6 && ext3 > 0.6 && ext4 > 0.6) score = 0.95;
          }} else if (currentTarget === "C") {{
            score = 0.88;
          }} else if (currentTarget === "D") {{
            if (ext1 > 0.6 && ext2 < 0.45 && ext3 < 0.45 && ext4 < 0.45) score = 0.95;
          }} else if (currentTarget === "I") {{
            if (ext4 > 0.55 && ext1 < 0.45 && ext2 < 0.45 && ext3 < 0.45) score = 0.95;
          }} else if (currentTarget === "L") {{
            if (ext1 > 0.55 && ext2 < 0.4 && ext3 < 0.4 && ext4 < 0.4) score = 0.95;
          }} else if (["O", "U", "V", "W"].includes(currentTarget)) {{
            if (ext1 > 0.5) score = 0.92;
          }}

          return Math.min(Math.max(score, 0.15), 0.98);
        }}

        function onResults(results) {{
          canvasCtx.save();
          canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

          let score = 0.15;
          if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {{
            for (const landmarks of results.multiHandLandmarks) {{
              drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {{color: '#00FF00', lineWidth: 4}});
              drawLandmarks(canvasCtx, landmarks, {{color: '#FF0000', fillColor: '#00FF00', lineWidth: 2, radius: 6}});
            }}
            score = evaluateGesture(results.multiHandLandmarks[0]);
          }}

          const pct = Math.round(score * 100);
          fillBar.style.width = pct + '%';
          fillBar.style.background = score >= 0.85 ? '#00ff00' : (score >= 0.6 ? '#ffff00' : '#ff4b4b');
          statusText.innerText = score >= 0.85 ? `MATCHED: '${{currentTarget}}' (${{pct}}%)` : `Match Score: ${{pct}}%`;

          canvasCtx.restore();
        }}

        const hands = new Hands({{
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${{file}}`
        }});

        hands.setOptions({{
          maxNumHands: 1,
          modelComplexity: 1,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        }});

        hands.onResults(onResults);

        const camera = new Camera(videoElement, {{
          onFrame: async () => {{
            await hands.send({{image: videoElement}});
          }},
          width: 640,
          height: 480
        }});

        camera.start().then(() => {{
          statusText.innerText = "Match Score: 0%";
        }});
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=520)
