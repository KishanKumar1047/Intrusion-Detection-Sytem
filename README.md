# Smart Door — Intrusion Detection System (Streamlit)

A face-recognition based smart door system with a web UI:

- **Left sidebar** — navigate between Live Detection, Registered Faces, Add New Face, Delete Face
- **Right side / main area** — live camera feed with face detection (on the Live Detection page)
- Recognized face → green box, "Entry Permitted"
- Unknown face → red box, "Alert"

## Why this version has no `dlib`

Earlier versions used the `face_recognition` library, which depends on `dlib` — a
C++ library that must be **compiled from source** on most cloud platforms
(Streamlit Cloud, Render, etc.). That build is memory-heavy and fails
unpredictably on free-tier hosting (out-of-memory kills, missing build tools,
Python-version mismatches).

This version instead uses **OpenCV's own built-in face models**:
- **YuNet** — face detection
- **SFace** — face recognition (turns a face into a 128-d vector, same idea as before)

Both ship as small ONNX files that OpenCV's DNN module loads directly.
**No C++ compiler, no cmake, no build tools, no `dlib` — anywhere.** This is
what makes the app reliably deployable on Streamlit Community Cloud (or Render,
or any other host) without native-build failures.

The two model files (~35 MB total) are downloaded automatically and cached to
disk the first time the app runs (see `utils/face_engine.py`). This needs an
internet connection on first run only — normal for any cloud deployment.

> **Note:** Because the recognition engine changed, any faces registered with
> the *old* `dlib`-based version are **not** compatible with this version.
> Simply register everyone again using **Add New Face** — it takes a few
> seconds per person.

## File Structure

```
smart_door_streamlit/
├── streamlit_app.py      # Main app: sidebar navigation + detection pages
├── utils/
│   ├── face_db.py        # Load / save the face-encodings database
│   └── face_engine.py    # YuNet + SFace wrapper (detection, encoding, matching)
├── requirements.txt       # Python dependencies (just streamlit, opencv, numpy, pillow)
├── runtime.txt             # Pins the Python version for Streamlit Cloud
└── data/
    └── encodings.pkl      # Auto-generated - stores all registered face encodings
```

(A `models/` folder is created automatically on first run to cache the
downloaded ONNX files — you don't need to create it yourself.)

## Run Locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (No dlib, no build tools needed — this should install cleanly on any
   Python 3.9+ environment.)

2. Run the app:
   ```
   streamlit run streamlit_app.py
   ```
   This opens the app in your browser (usually at `http://localhost:8501`).
   The first time you visit the app, it downloads the two model files
   (~35 MB) — you'll see a "Loading face recognition models..." spinner
   once, then it's cached for future runs.

## Deploy to Streamlit Community Cloud (free)

1. Push this project to a **public GitHub repository**.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"**, select your repo/branch, and set the main file to
   `streamlit_app.py`.
4. Click **Deploy**. Since there's no native build step anymore, this should
   complete in under a minute.

## How to Use

1. **Add New Face** — enter a name, then either take a photo with your webcam or
   upload a few images (3-5 samples from different angles improves accuracy).
2. **Registered Faces** — see everyone currently registered and how many samples
   each person has.
3. **Delete Face** — remove a person from the database.
4. **Live Detection** — camera preview is live; click "Take Photo" whenever
   someone is at the door. Recognized → green box + "Entry Permitted".
   Unknown → red box + "Alert".

## Tuning

- `MATCH_THRESHOLD` in `utils/face_engine.py` (default `0.363`, OpenCV's
  recommended value for SFace) controls matching strictness. Higher = stricter.
