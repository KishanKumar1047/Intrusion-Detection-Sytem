"""
face_engine.py
---------------
Wraps OpenCV's built-in YuNet (face detection) and SFace (face recognition)
DNN models. These ship as small ONNX files and load directly through
OpenCV's DNN module - no dlib, no C++ compiling, no cmake/build-tools
required anywhere. This is what makes the app deployable on Streamlit
Cloud (or Render, or anywhere) without native-build failures.

The two model files are downloaded once and cached to disk under
models/ on first run (they're a few hundred KB and ~35MB respectively).
"""

import os
import urllib.request

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Multiple mirrors per file - if the first fails, the next is tried.
MODEL_URLS = {
    "face_detection_yunet.onnx": [
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ],
    "face_recognition_sface.onnx": [
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ],
}

# OpenCV Zoo's recommended cosine-similarity threshold for SFace: scores at or
# above this are considered the same person. Higher score = more similar.
MATCH_THRESHOLD = 0.363


def _ensure_model(filename):
    """Downloads a model file into MODEL_DIR if it isn't already cached there."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    dest = os.path.join(MODEL_DIR, filename)

    # A valid model file is at least a few hundred KB; anything smaller
    # (e.g. a failed/partial download) is treated as missing.
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        return dest

    last_error = None
    for url in MODEL_URLS[filename]:
        try:
            urllib.request.urlretrieve(url, dest)
            if os.path.getsize(dest) > 10_000:
                return dest
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Could not download required model '{filename}': {last_error}")


class FaceEngine:
    """Loads YuNet + SFace once and exposes simple detect / encode / compare helpers."""

    def __init__(self):
        yunet_path = _ensure_model("face_detection_yunet.onnx")
        sface_path = _ensure_model("face_recognition_sface.onnx")

        self.detector = cv2.FaceDetectorYN.create(
            yunet_path, "", (320, 320), score_threshold=0.7
        )
        self.recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    def detect_faces(self, image_bgr):
        """Runs YuNet on a BGR image. Returns an array of detections, one row per
        face: [x, y, w, h, <5 landmark x/y pairs>, confidence_score]. Empty list
        if no faces found."""
        h, w = image_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image_bgr)
        return faces if faces is not None else []

    def get_encoding(self, image_bgr, face_row):
        """Aligns + crops a detected face using its landmarks, then returns its
        128-d SFace feature vector (shape (1, 128), float32)."""
        aligned = self.recognizer.alignCrop(image_bgr, face_row)
        return self.recognizer.feature(aligned)

    def compare(self, feature1, feature2):
        """Cosine similarity between two SFace feature vectors (higher = more similar)."""
        return self.recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
