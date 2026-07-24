"""
streamlit_app.py
-----------------
Smart Door - Intrusion Detection System (Streamlit Web App)

Layout:
  - Left sidebar : navigation (Live Detection / Registered Faces / Add New Face / Delete Face)
  - Right side    : continuous live camera feed with real-time face detection

Uses OpenCV's built-in YuNet (detection) + SFace (recognition) models for
recognition - no dlib, no C++ compiling. Uses streamlit-webrtc + av for the
continuous live video stream - both install from prebuilt wheels (av bundles
FFmpeg statically), so nothing here needs to compile from source. This
combination deploys cleanly on Streamlit Cloud.

Run locally with:
    streamlit run streamlit_app.py
"""

import threading

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from utils.face_db import load_database, save_database
from utils.face_engine import FaceEngine, MATCH_THRESHOLD

st.set_page_config(page_title="Smart Door - Intrusion Detection", layout="wide")


# ---------------------------------------------------------------------------
# Cached engine loader - downloads/loads the models only once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading face recognition models (first run only)...")
def get_engine():
    return FaceEngine()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_known_list(db):
    """Flattens the {name: [encodings]} database into two parallel lists
    for fast comparison against a live frame."""
    names, encs = [], []
    for name, enc_list in db.items():
        for e in enc_list:
            names.append(name)
            encs.append(e)
    return names, encs


def best_match(engine, feature, known_names, known_encs):
    """Compares one face's feature vector against every stored encoding.
    Returns (name, score). name is 'Unknown' if nothing clears the threshold."""
    best_name = "Unknown"
    best_score = -1.0
    for name, enc in zip(known_names, known_encs):
        score = engine.compare(feature, enc)
        if score > best_score:
            best_score = score
            if score >= MATCH_THRESHOLD:
                best_name = name
    return best_name, best_score


def annotate_frame(engine, img_bgr, known_names, known_encs):
    """Runs detection + recognition on a BGR frame and draws boxes/labels
    directly onto it. Returns the annotated frame plus a list of results."""
    faces = engine.detect_faces(img_bgr)
    results = []


    for face_row in faces:
        x, y, w, h = face_row[0:4].astype(int)
        feature = engine.get_encoding(img_bgr, face_row)
        name, score = best_match(engine, feature, known_names, known_encs)

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 2)
        label = "Entry Permitted" if name != "Unknown" else "Unknown - Alert"
        cv2.putText(
            img_bgr, f"{name}: {label}", (x, max(y - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
        results.append((name, label))

    return img_bgr, results


def encode_uploaded_image(engine, pil_image):
    """Detects the first face in a PIL image and returns its feature vector,
    or None if no face was found."""
    image_np = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    faces = engine.detect_faces(img_bgr)
    if len(faces) == 0:
        return None
    return engine.get_encoding(img_bgr, faces[0])


class DoorVideoProcessor(VideoProcessorBase):
    """Receives each live webcam frame from streamlit-webrtc and draws
    recognition results directly onto it in real time."""

    def __init__(self, engine, known_names, known_encs):
        self.engine = engine
        self.known_names = known_names
        self.known_encs = known_encs
        self.lock = threading.Lock()
        self.last_results = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        annotated, results = annotate_frame(self.engine, img, self.known_names, self.known_encs)
        with self.lock:
            self.last_results = results
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Smart Door Control Panel")
page = st.sidebar.radio(
    "Navigation",
    ["Live Detection", "Registered Faces", "Add New Face", "Delete Face"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Face Recognition Intrusion Detection System")

db_preview = load_database()
st.sidebar.metric("Registered People", len(db_preview))

engine = get_engine()


# ---------------------------------------------------------------------------
# Page: Live Detection
# ---------------------------------------------------------------------------
if page == "Live Detection":
    st.title("Smart Door - Live Intrusion Detection")
    st.write(
        "The feed below continuously scans for faces. Green box = recognized "
        "person (Entry Permitted). Red box = unknown person (Alert)."
    )

    db = load_database()
    known_names, known_encs = build_known_list(db)

    if not db:
        st.warning("No faces registered yet. Go to 'Add New Face' first, so the detector has someone to recognize.")

    ctx = webrtc_streamer(
        key="door-detection",
        video_processor_factory=lambda: DoorVideoProcessor(engine, known_names, known_encs),
        media_stream_constraints={"video": True, "audio": False},
    )

    st.info(
        "If you just registered or deleted a face, click 'Stop' then 'Start' "
        "again (or refresh the page) so the detector picks up the latest database."
    )


# ---------------------------------------------------------------------------
# Page: Registered Faces
# ---------------------------------------------------------------------------
elif page == "Registered Faces":
    st.title("Registered Faces")
    db = load_database()

    if not db:
        st.warning("No faces registered yet. Go to 'Add New Face' to register someone.")
    else:
        for name, encs in db.items():
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(name)
                with col2:
                    st.metric("Samples", len(encs))


# ---------------------------------------------------------------------------
# Page: Add New Face
# ---------------------------------------------------------------------------
elif page == "Add New Face":
    st.title("Add New Face")
    name = st.text_input("Person's name")

    tab1, tab2 = st.tabs(["Capture from Webcam", "Upload Images"])

    with tab1:
        img_file = st.camera_input("Take a photo")
        if img_file is not None:
            if not name:
                st.warning("Please enter a name before registering.")
            elif st.button("Register this photo", key="register_webcam"):
                image = Image.open(img_file)
                feature = encode_uploaded_image(engine, image)

                if feature is None:
                    st.error("No face detected in the photo. Please try again.")
                else:
                    db = load_database()
                    existing = db.get(name, [])
                    db[name] = existing + [feature]
                    save_database(db)
                    st.success(f"'{name}' registered successfully!")

    with tab2:
        uploaded_files = st.file_uploader(
            "Upload one or more photos (3-5 recommended, different angles/lighting)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            if not name:
                st.warning("Please enter a name before registering.")
            elif st.button("Register uploaded photos", key="register_upload"):
                new_encodings = []
                for file in uploaded_files:
                    image = Image.open(file)
                    feature = encode_uploaded_image(engine, image)
                    if feature is None:
                        st.warning(f"No face found in {file.name}, skipped.")
                        continue
                    new_encodings.append(feature)

                if new_encodings:
                    db = load_database()
                    existing = db.get(name, [])
                    db[name] = existing + new_encodings
                    save_database(db)
                    st.success(f"'{name}' registered with {len(new_encodings)} sample(s)!")
                else:
                    st.error("No valid faces found in the uploaded photos.")


# ---------------------------------------------------------------------------
# Page: Delete Face
# ---------------------------------------------------------------------------
elif page == "Delete Face":
    st.title("Delete a Registered Face")
    db = load_database()
    names = list(db.keys())

    if not names:
        st.info("No faces registered.")
    else:
        target = st.selectbox("Select a person to delete", names)
        if st.button("Delete", type="primary"):
            del db[target]
            save_database(db)
            st.success(f"'{target}' deleted successfully.")
            st.rerun()
