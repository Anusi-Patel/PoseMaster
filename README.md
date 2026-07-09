# 🕺 PoseMaster

PoseMaster is a real-time, webcam-based pose-matching game. It shows you a
target pose, tracks your body with your webcam, and scores how closely you
match it — built on [MediaPipe](https://developers.google.com/mediapipe) pose
detection and [OpenCV](https://opencv.org/).

## How it works

1. **`pose_extractor.py`** builds a pose library: it reads images from the
   `poses/` folder, runs MediaPipe pose detection on each one, computes 8 key
   joint angles (elbows, shoulders, knees, hips) plus a normalized skeleton
   silhouette, and saves it all to `pose_library.json`.
2. **`posemaster_game.py`** is the actual game: it loads the pose library,
   picks poses at random, opens your webcam, overlays the target pose's
   skeleton in the corner, and scores your live pose against it in real time
   using a joint-angle comparison — full points if you nail the angle, fewer
   the further off you are.
3. **`demo.py`** is a simpler single-pose sandbox: press `S` to save whatever
   pose you're currently in as the target, then try to match it live, with no
   pose library or rounds needed. Good for quickly testing the scoring math.

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/<your-username>/PoseMaster.git
cd PoseMaster
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Build the pose library** (only needed once, or whenever you add new
images to `poses/`)

```bash
python pose_extractor.py
```

This reads every `.jpg` / `.jpeg` / `.png` / `.webp` file in `poses/` and
regenerates `pose_library.json`.

**3. Play**

```bash
python posemaster_game.py
```

Or try the simpler sandbox instead:

```bash
python demo.py
```

## Controls

**`posemaster_game.py`**
- Match the target pose shown in the corner before the timer runs out
- `Q` — quit anytime

**`demo.py`**
- `S` — save your current pose as the target
- `R` — reset the target pose
- `Q` — quit

## Project structure

```
posemaster_game.py     # the main game
pose_extractor.py       # builds pose_library.json from images in poses/
demo.py                 # simple single-pose sandbox / scoring test
pose_library.json       # generated pose data (angles + silhouette per pose)
poses/                   # source images used to build the pose library
poses_extracted/         # extracted/processed copies of the pose images
requirements.txt
```

## Tech stack

- [OpenCV](https://opencv.org/) — webcam capture and rendering
- [MediaPipe](https://developers.google.com/mediapipe) — pose landmark detection
- [NumPy](https://numpy.org/) — angle math

## Notes

- Requires a working webcam.
- `pose_library.json` currently stores image paths in Windows format
  (`poses\\pose.webp`); if you rebuild it on macOS/Linux with
  `pose_extractor.py`, paths will be regenerated in the correct format for
  your OS.
- This is a personal/portfolio project, not hardened for production use.

## License

Add a license of your choice (e.g. MIT) if you want others to be able to
reuse this code — see [choosealicense.com](https://choosealicense.com/).
