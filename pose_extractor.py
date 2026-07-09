"""
PoseMaster — Pose Extractor
Run this ONCE to build your pose library from images.

Usage:
    1. Put all pose images in a folder called 'poses/'
    2. Run: python pose_extractor.py
    3. It creates 'pose_library.json' — your game loads from this

Supported formats: .jpg .jpeg .png .webp
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
from pathlib import Path

# ── SAME ANGLE MATH AS STAGE 2 ────────────────────────────────
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

def extract_key_angles(landmarks):
    def lm(idx):
        return [landmarks[idx].x, landmarks[idx].y]
    return {
        "left_elbow":     calculate_angle(lm(11), lm(13), lm(15)),
        "right_elbow":    calculate_angle(lm(12), lm(14), lm(16)),
        "left_shoulder":  calculate_angle(lm(13), lm(11), lm(23)),
        "right_shoulder": calculate_angle(lm(14), lm(12), lm(24)),
        "left_knee":      calculate_angle(lm(23), lm(25), lm(27)),
        "right_knee":     calculate_angle(lm(24), lm(26), lm(28)),
        "left_hip":       calculate_angle(lm(11), lm(23), lm(25)),
        "right_hip":      calculate_angle(lm(12), lm(24), lm(26)),
    }

def extract_landmarks_for_silhouette(landmarks, img_w, img_h):
    """
    Save raw landmark positions so we can DRAW the silhouette in-game.
    Returns a dict of landmark_index -> [x_pixel, y_pixel]
    """
    key_indices = [
        0,   # nose
        11, 12,  # shoulders
        13, 14,  # elbows
        15, 16,  # wrists
        23, 24,  # hips
        25, 26,  # knees
        27, 28,  # ankles
    ]
    positions = {}
    for idx in key_indices:
        lm = landmarks[idx]
        positions[str(idx)] = [
            round(lm.x, 4),
            round(lm.y, 4)
        ]
    return positions

# ── MEDIAPIPE SETUP ───────────────────────────────────────────
mp_pose = mp.solutions.pose
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}

# ── PROCESS ALL IMAGES ────────────────────────────────────────
poses_dir = Path("poses")
if not poses_dir.exists():
    print("❌ 'poses/' folder not found. Create it and add your images.")
    exit()

image_files = [
    f for f in sorted(poses_dir.iterdir())
    if f.suffix.lower() in SUPPORTED
]

if not image_files:
    print(f"❌ No supported images found in 'poses/'. Add .jpg/.png/.webp files.")
    exit()

print(f"Found {len(image_files)} images. Processing...\n")

pose_library = []

with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    for img_path in image_files:
        print(f"Processing: {img_path.name} ... ", end="")

        img = cv2.imread(str(img_path))
        if img is None:
            # Try reading webp differently
            img = cv2.imdecode(
                np.frombuffer(open(img_path, 'rb').read(), np.uint8),
                cv2.IMREAD_COLOR
            )

        if img is None:
            print("❌ Could not read image, skipping.")
            continue

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not results.pose_landmarks:
            print("⚠️  No person detected, skipping.")
            continue

        landmarks = results.pose_landmarks.landmark
        angles = extract_key_angles(landmarks)
        silhouette = extract_landmarks_for_silhouette(landmarks, w, h)

        pose_entry = {
            "name": img_path.stem,           # filename without extension
            "image": str(img_path),           # path to original image
            "angles": angles,
            "silhouette": silhouette,
            "difficulty": 1                   # 1=easy, 2=medium, 3=hard (edit manually later)
        }

        pose_library.append(pose_entry)
        print(f"✅ Saved ({len(angles)} angles)")

        # Show a preview window so you can verify it detected correctly
        mp_draw = mp.solutions.drawing_utils
        preview = img.copy()
        mp_draw.draw_landmarks(
            preview,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        # Resize preview to fit screen
        preview_h = 400
        preview_w = int(w * (preview_h / h))
        preview = cv2.resize(preview, (preview_w, preview_h))
        # Save skeleton overlay image
        output_img_dir = Path("poses_extracted")
        output_img_dir.mkdir(exist_ok=True)
        save_path = output_img_dir / img_path.name
        cv2.imwrite(str(save_path), preview)
        print(f"   Saved skeleton image → {save_path}")

# Show preview
cv2.imshow(f"Preview: {img_path.name} (any key = next)", preview)
cv2.waitKey(0)

cv2.destroyAllWindows()

# ── SAVE LIBRARY ──────────────────────────────────────────────
if not pose_library:
    print("\n❌ No poses extracted. Check your images.")
    exit()

output_path = Path("pose_library.json")
with open(output_path, "w") as f:
    json.dump(pose_library, f, indent=2)

print(f"\n{'='*50}")
print(f"✅ Pose library saved: {output_path}")
print(f"   Total poses: {len(pose_library)}")
print(f"\nPoses extracted:")
for i, p in enumerate(pose_library, 1):
    print(f"  {i}. {p['name']}")
print(f"\nNext step: run posemaster_game.py")
print(f"{'='*50}")