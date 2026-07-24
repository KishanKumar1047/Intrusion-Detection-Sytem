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

import cv2
import numpy as np
import streamlit as st
import face_recognition
from PIL import Image

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


def run_detection_on_frame(image_np, known_names, known_encs):
    """Runs face detection + recognition on a single RGB image (numpy array).
    Draws bounding boxes/labels and returns the annotated RGB image plus a
    list of (name, label) results."""
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)

    results = []

    for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
        name = "Unknown"
        if known_encs:
            distances = face_recognition.face_distance(known_encs, face_enc)
            best_idx = int(np.argmin(distances))
            if distances[best_idx] <= TOLERANCE:
                name = known_names[best_idx]

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(img_bgr, (left, top), (right, bottom), color, 2)
        label = "Entry Permitted" if name != "Unknown" else "Unknown - Alert"
        cv2.putText(
            img_bgr, f"{name}: {label}", (left, max(top - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
        results.append((name, label))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, results


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
    st.title("Smart Door - Live Intrusion Detection")
    st.write(
        "The camera preview below is live. Click 'Take Photo' whenever someone "
        "is at the door to scan them. Green box = recognized person (Entry "
        "Permitted). Red box = unknown person (Alert)."
    )

    db = load_database()
    known_names, known_encs = build_known_list(db)

    frame = st.camera_input("Door Camera")

    if frame is not None:
        image = Image.open(frame)
        image_np = np.array(image.convert("RGB"))
        annotated_rgb, results = run_detection_on_frame(image_np, known_names, known_encs)

        st.image(annotated_rgb, caption="Detection Result", use_container_width=True)

        if not results:
            st.warning("No face detected in the frame.")
        else:
            for name, label in results:
                if name != "Unknown":
                    st.success(f"Recognized - {name}: {label}")
                else:
                    st.error(f"Alert - {name}: {label}")


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
