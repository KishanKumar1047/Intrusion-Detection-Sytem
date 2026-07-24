"""
streamlit_app.py
-----------------
Smart Door - Intrusion Detection System (Streamlit Web App)

Layout:
  - Left sidebar : navigation (Live Detection / Registered Faces / Add New Face / Delete Face)
  - Right side    : live camera feed with real-time face detection (on the Live Detection page)

Run locally with:
    streamlit run streamlit_app.py
"""

import threading

import av
import cv2
import numpy as np
import streamlit as st
import face_recognition
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from utils.face_db import load_database, save_database

TOLERANCE = 0.50  # lower = stricter matching (0.4-0.6 is a reasonable range)

st.set_page_config(page_title="Smart Door - Intrusion Detection", layout="wide")


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


class FaceRecognitionProcessor(VideoProcessorBase):
    """Receives each webcam frame, runs face recognition, and draws
    bounding boxes + labels directly on the frame before it's displayed."""

    def __init__(self):
        db = load_database()
        self.known_names, self.known_encs = build_known_list(db)
        self.lock = threading.Lock()
        self.last_status = "Scanning..."

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Downscale for faster processing
        small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        status_texts = []

        for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
            name = "Unknown"
            if self.known_encs:
                distances = face_recognition.face_distance(self.known_encs, face_enc)
                best_idx = int(np.argmin(distances))
                if distances[best_idx] <= TOLERANCE:
                    name = self.known_names[best_idx]

            # scale coordinates back up to the original frame size
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            label = "Entry Permitted" if name != "Unknown" else "Unknown - Alert"
            cv2.putText(
                img, f"{name}: {label}", (left, max(top - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
            status_texts.append(f"{name}: {label}")

        with self.lock:
            self.last_status = "; ".join(status_texts) if status_texts else "No face detected"

        return av.VideoFrame.from_ndarray(img, format="bgr24")


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


# ---------------------------------------------------------------------------
# Page: Live Detection
# ---------------------------------------------------------------------------
if page == "Live Detection":
    st.title("🚪 Smart Door — Live Intrusion Detection")
    st.write(
        "The camera feed below continuously scans for faces. "
        "**Green box** = recognized person (Entry Permitted). "
        "**Red box** = unknown person (Alert)."
    )

    webrtc_streamer(
        key="door-detection",
        video_processor_factory=FaceRecognitionProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    st.info(
        "Tip: If you just registered or deleted a face, click **Start** again "
        "(or refresh the page) so the detector picks up the latest database."
    )


# ---------------------------------------------------------------------------
# Page: Registered Faces
# ---------------------------------------------------------------------------
elif page == "Registered Faces":
    st.title("👥 Registered Faces")
    db = load_database()

    if not db:
        st.warning("No faces registered yet. Go to **Add New Face** to register someone.")
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
    st.title("➕ Add New Face")
    name = st.text_input("Person's name")

    tab1, tab2 = st.tabs(["📷 Capture from Webcam", "📁 Upload Images"])

    with tab1:
        img_file = st.camera_input("Take a photo")
        if img_file is not None:
            if not name:
                st.warning("Please enter a name before registering.")
            elif st.button("Register this photo", key="register_webcam"):
                image = Image.open(img_file)
                image_np = np.array(image.convert("RGB"))
                face_locations = face_recognition.face_locations(image_np)

                if not face_locations:
                    st.error("No face detected in the photo. Please try again.")
                else:
                    encodings = face_recognition.face_encodings(image_np, face_locations)
                    db = load_database()
                    existing = db.get(name, [])
                    db[name] = existing + [encodings[0]]
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
                    image_np = np.array(image.convert("RGB"))
                    face_locations = face_recognition.face_locations(image_np)
                    if not face_locations:
                        st.warning(f"No face found in {file.name}, skipped.")
                        continue
                    encs = face_recognition.face_encodings(image_np, face_locations)
                    new_encodings.append(encs[0])

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
    st.title("🗑️ Delete a Registered Face")
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
