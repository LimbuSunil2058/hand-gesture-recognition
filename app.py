import streamlit as st
import cv2
import torch
import timm
import mediapipe as mp
import urllib.request
from torchvision import transforms
from PIL import Image
import os

st.set_page_config(
    page_title="Hand Gesture Recognition",
    layout="wide"
)

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False
if "cap" not in st.session_state:
    st.session_state.cap = None

@st.cache_resource
def load_mediapipe():
    if not os.path.exists("hand_landmarker.task"):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            "hand_landmarker.task"
        )
    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE
    )
    return HandLandmarker.create_from_options(options)

@st.cache_resource
def load_model():
    device     = torch.device("cpu")
    model      = timm.create_model("efficientnet_b0", pretrained=False, num_classes=7)
    checkpoint = torch.load("best_model_hand_detection.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, device

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

classes   = ["call", "dislike", "fist", "like", "mute", "ok", "stop"]
emoji_map = {
    "call"   : "🤙",
    "dislike": "👎",
    "fist"   : "✊",
    "like"   : "👍",
    "mute"   : "🤫",
    "ok"     : "👌",
    "stop"   : "✋"
}
description_map = {
    "call"   : "Extend thumb & pinky, curl other fingers",
    "dislike": "Point thumb downward",
    "fist"   : "Close all fingers into a fist",
    "like"   : "Point thumb upward",
    "mute"   : "Place index finger on lips",
    "ok"     : "Connect thumb & index into a circle",
    "stop"   : "Open palm facing forward"
}

def crop_hand(image, landmarker):
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )
    result = landmarker.detect(mp_image)
    if result.hand_landmarks:
        h, w = image.shape[:2]
        lm = result.hand_landmarks[0]
        x_coords = [l.x * w for l in lm]
        y_coords = [l.y * h for l in lm]
        pad  = 10
        x1   = max(0, int(min(x_coords)) - pad)
        y1   = max(0, int(min(y_coords)) - pad)
        x2   = min(w, int(max(x_coords)) + pad)
        y2   = min(h, int(max(y_coords)) + pad)
        side = max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, cx + side // 2)
        y2 = min(h, cy + side // 2)
        return image[y1:y2, x1:x2], (x1, y1, x2, y2)
    return None, None

def predict(cropped_img, model, device):
    img    = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs    = model(tensor)
        probs      = torch.softmax(outputs, dim=1)
        conf, pred = probs.max(1)
    return classes[pred.item()], conf.item()

def release_camera():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    cv2.destroyAllWindows()
    st.session_state.camera_on = False

# LOAD
with st.spinner("Loading models..."):
    landmarker    = load_mediapipe()
    model, device = load_model()

# LAYOUT
st.title("Hand Gesture Recognition System")
st.markdown("---")

# Gesture Guide
st.subheader("Gesture Guide")
st.caption("Show any of these 7 gestures to the camera:")

cols = st.columns(7)
for i, cls in enumerate(classes):
    with cols[i]:
        st.markdown(
            f"""
            <div style='text-align:center; padding:10px;
                        border:1px solid #444; border-radius:10px;'>
                <div style='font-size:2rem'>{emoji_map[cls]}</div>
                <div style='font-weight:bold; margin-top:6px'>{cls.upper()}</div>
                <div style='font-size:0.75rem; color:gray;
                            margin-top:4px'>{description_map[cls]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")

# Camera Controls
st.subheader("📷 Live Camera")
col_start, col_stop, col_note = st.columns([1, 1, 4])

with col_start:
    if st.button("▶️ Start Camera", disabled=st.session_state.camera_on):
        st.session_state.camera_on = True
        st.rerun()

with col_stop:
    if st.button("⏹️ Stop Camera", disabled=not st.session_state.camera_on):
        release_camera()
        st.rerun()

with col_note:
    if st.session_state.camera_on:
        st.info("Click **Stop Camera** to turn off the camera.")

# Camera Feed
if st.session_state.camera_on:
    col1, col2 = st.columns([2, 1])

    with col1:
        frame_window = st.image([])
    with col2:
        st.markdown("### Prediction")
        prediction_box = st.empty()
        confidence_box = st.empty()
        st.markdown("---")

    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)

    cap = st.session_state.cap

    if not cap.isOpened():
        st.error("Camera not found! Check your camera connection.")
        release_camera()
    else:
        while st.session_state.camera_on:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera!")
                release_camera()
                break

            frame   = cv2.flip(frame, 1)
            cropped, bbox = crop_hand(frame, landmarker)

            if cropped is not None and cropped.size > 0:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label, confidence = predict(cropped, model, device)
                emoji = emoji_map[label]

                if confidence > 0.93:
                    # Show label on frame
                    cv2.putText(frame, f"{label.upper()} {confidence:.0%}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), 2)
                    prediction_box.markdown(
                        f"<div style='text-align:center'>"
                        f"<div style='font-size:4rem'>{emoji}</div>"
                        f"<div style='font-size:1.8rem; font-weight:bold'>"
                        f"{label.upper()}</div></div>",
                        unsafe_allow_html=True
                    )
                    confidence_box.progress(
                        confidence, text=f"Confidence: {confidence:.1%}"
                    )
                else:
                    # Show "Not sure" on frame too
                    cv2.putText(frame, "Not sure...",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 255), 2)
                    prediction_box.markdown("**Not sure...**")
                    confidence_box.empty()
            else:
                prediction_box.markdown(
                    "<div style='text-align:center; color:gray'>"
                    "👀<br>No hand detected</div>",
                    unsafe_allow_html=True
                )
                confidence_box.empty()
                cv2.putText(frame, "No hand detected",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb, channels="RGB")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                release_camera()
                st.rerun()
                break

else:
    pass