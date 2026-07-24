# Smart Door — Intrusion Detection System (Streamlit)

A face-recognition based smart door system with a web UI:

- **Left sidebar** — navigate between Live Detection, Registered Faces, Add New Face, Delete Face
- **Right side** — live camera feed with real-time face detection (on the Live Detection page)
- Recognized face → green box, "Entry Permitted"
- Unknown face → red box, "Alert"

## 📁 File Structure

```
smart_door_streamlit/
├── streamlit_app.py      # Main app: sidebar navigation + live detection
├── utils/
│   └── face_db.py        # Load/save the face-encodings database
├── data/
│   └── encodings.pkl     # Auto-generated - stores all registered face encodings
├── requirements.txt       # Python dependencies
├── packages.txt           # System packages (needed to build dlib on Streamlit Cloud)
└── runtime.txt            # Pins Python version for Streamlit Cloud
```

## ⚙️ Run Locally

1. Install Python 3.10 or 3.11 (recommended — newer versions can hit `dlib`/`setuptools`
   build issues, as you've already seen).

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   > `face_recognition` depends on `dlib`, which needs a C++ compiler to build from
   > source. On Windows, either:
   > - Install a precompiled wheel from https://github.com/z-mahmud22/Dlib_Windows_Python3.x
   >   matching your Python version, **or**
   > - Install CMake + Visual Studio Build Tools ("Desktop development with C++")

3. Run the app:
   ```
   streamlit run streamlit_app.py
   ```
   This opens the app in your browser (usually at `http://localhost:8501`). Your
   browser will ask for camera permission on the Live Detection page.

## 🚀 Deploy to Streamlit Community Cloud (free)

1. Push this project to a **public GitHub repository**.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"**, select your repo, branch, and set the main file to
   `streamlit_app.py`.
4. Streamlit Cloud will automatically:
   - Use `runtime.txt` to pick the Python version
   - Install OS packages from `packages.txt` (needed to compile `dlib`)
   - Install Python packages from `requirements.txt`
5. Click **Deploy**. First build can take 5-10 minutes since `dlib` compiles from source.

> Note: On Streamlit Community Cloud, the live camera feed uses the **viewer's own
> webcam** (via their browser, through `streamlit-webrtc`) — not a server-side camera.
> This is the correct setup for a demo/dashboard; for an actual physical door, you'd
> run this on a device with a camera attached (e.g. a Raspberry Pi) instead.

## 🧭 How to Use

1. **Add New Face** — enter a name, then either take a photo with your webcam or
   upload a few images (3-5 samples from different angles improves accuracy).
2. **Registered Faces** — see everyone currently registered and how many samples
   each person has.
3. **Delete Face** — remove a person from the database.
4. **Live Detection** — starts the webcam feed; faces are boxed and labeled in
   real time as either recognized ("Entry Permitted") or unknown ("Alert").

## 🔧 Tuning

- `TOLERANCE` in `streamlit_app.py` (default `0.50`) controls matching strictness.
  Lower = stricter (fewer false positives), higher = looser.
- If you add or delete a face while the Live Detection page is already running,
  click **Start** again (or refresh the page) so the new database is loaded.
