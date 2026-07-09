"""
PoseMaster — The Game
Match the pose before the timer runs out!

Controls: Q = quit anytime
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import random
from pathlib import Path

# ── SETTINGS ──────────────────────────────────────────────────
POSE_HOLD_SECONDS   = 5      # how long player has to match pose
COUNTDOWN_SECONDS   = 3      # countdown before scoring starts
WIN_THRESHOLD       = 75     # score needed to "pass" a pose
TOTAL_ROUNDS        = 5      # how many poses per game
POSES_FILE          = "pose_library.json"

# ── COLOURS (BGR) ─────────────────────────────────────────────
WHITE   = (255, 255, 255)
BLACK   = (0,   0,   0)
GREEN   = (0,   255, 0)
RED     = (0,   0,   255)
YELLOW  = (0,   200, 255)
CYAN    = (255, 255, 0)
PURPLE  = (255, 0,   200)

# ── MEDIAPIPE ─────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ── ANGLE MATH ────────────────────────────────────────────────
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

def score_pose(target, player, threshold=30):
    scores = []
    for joint in target:
        diff = abs(target[joint] - player[joint])
        scores.append(max(0, 100 - (diff / threshold * 100)))
    return round(sum(scores) / len(scores), 1)

# ── SILHOUETTE DRAWING ────────────────────────────────────────
SKELETON_CONNECTIONS = [
    ("11", "12"),  # shoulders
    ("11", "13"), ("13", "15"),  # left arm
    ("12", "14"), ("14", "16"),  # right arm
    ("11", "23"), ("12", "24"),  # torso sides
    ("23", "24"),                # hips
    ("23", "25"), ("25", "27"),  # left leg
    ("24", "26"), ("26", "28"),  # right leg
]

def draw_silhouette(canvas, silhouette, color=CYAN, alpha=0.6):
    """Draw target pose skeleton onto a canvas."""
    h, w = canvas.shape[:2]
    overlay = canvas.copy()

    # Draw connections
    for a, b in SKELETON_CONNECTIONS:
        if a in silhouette and b in silhouette:
            pt1 = (int(silhouette[a][0] * w), int(silhouette[a][1] * h))
            pt2 = (int(silhouette[b][0] * w), int(silhouette[b][1] * h))
            cv2.line(overlay, pt1, pt2, color, 4)

    # Draw joints
    for idx, (x, y) in silhouette.items():
        pt = (int(x * w), int(y * h))
        cv2.circle(overlay, pt, 8, color, -1)
        cv2.circle(overlay, pt, 8, WHITE, 2)

    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

# ── LOAD POSE LIBRARY ─────────────────────────────────────────
def load_poses():
    if not Path(POSES_FILE).exists():
        print(f"❌ {POSES_FILE} not found. Run pose_extractor.py first.")
        exit()
    with open(POSES_FILE) as f:
        poses = json.load(f)
    print(f"✅ Loaded {len(poses)} poses from library.")
    return poses

# ── SCREEN HELPERS ────────────────────────────────────────────
def draw_centered_text(frame, text, y, size=1.2, color=WHITE, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, size, thickness)
    x = (frame.shape[1] - tw) // 2
    # Shadow
    cv2.putText(frame, text, (x+2, y+2), font, size, BLACK, thickness+2)
    cv2.putText(frame, text, (x, y), font, size, color, thickness)

def draw_timer_bar(frame, elapsed, total, color=GREEN):
    h, w = frame.shape[:2]
    bar_w = w - 40
    filled = int(bar_w * max(0, (total - elapsed) / total))
    cv2.rectangle(frame, (20, h - 30), (20 + bar_w, h - 10), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, h - 30), (20 + filled, h - 10), color, -1)

def show_reference_image(frame, image_path, silhouette):
    """Show target pose in top-right corner with skeleton overlay."""
    h, w = frame.shape[:2]
    ref_w, ref_h = 200, 200

    try:
        ref = cv2.imread(image_path)
        if ref is None:
            ref = cv2.imdecode(
                np.frombuffer(open(image_path, 'rb').read(), np.uint8),
                cv2.IMREAD_COLOR
            )
        if ref is not None:
            ref = cv2.resize(ref, (ref_w, ref_h))
            draw_silhouette(ref, silhouette, color=CYAN)
            # Place in top-right corner with border
            x_offset = w - ref_w - 10
            y_offset = 10
            cv2.rectangle(frame,
                          (x_offset - 3, y_offset - 3),
                          (x_offset + ref_w + 3, y_offset + ref_h + 3),
                          CYAN, 2)
            frame[y_offset:y_offset+ref_h, x_offset:x_offset+ref_w] = ref
            cv2.putText(frame, "TARGET", (x_offset + 50, y_offset + ref_h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
    except Exception:
        pass

# ── GAME STATES ───────────────────────────────────────────────
STATE_INTRO     = "intro"
STATE_COUNTDOWN = "countdown"
STATE_PLAYING   = "playing"
STATE_RESULT    = "result"
STATE_GAMEOVER  = "gameover"

# ── MAIN GAME ─────────────────────────────────────────────────
def run_game():
    poses = load_poses()
    selected_poses = random.sample(poses, min(TOTAL_ROUNDS, len(poses)))

    cap = cv2.VideoCapture(0)
    cap.set(3, 800)
    cap.set(4, 600)

    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        return

    # Game variables
    state           = STATE_INTRO
    round_index     = 0
    current_pose    = None
    round_score     = 0
    total_score     = 0
    scores_history  = []
    state_start     = time.time()
    best_score      = 0
    flash_timer     = 0

    print("\n🎮 PoseMaster started! Press Q to quit.\n")

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose_detector:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror for natural feel
            h, w = frame.shape[:2]
            now = time.time()
            elapsed = now - state_start

            # Detect pose
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(rgb)
            player_angles = None

            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=GREEN, thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0, 180, 0), thickness=2)
                )
                player_angles = extract_key_angles(results.pose_landmarks.landmark)

            # ── Flash effect ──────────────────────────────────
            if flash_timer > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), GREEN, -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                flash_timer -= 1

            # ══════════════════════════════════════════════════
            # STATE: INTRO
            # ══════════════════════════════════════════════════
            if state == STATE_INTRO:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), BLACK, -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                draw_centered_text(frame, "POSEMASTER", h//2 - 80, size=2.5, color=CYAN)
                draw_centered_text(frame, "Match the pose before time runs out!", h//2, size=0.7, color=WHITE)
                draw_centered_text(frame, f"{TOTAL_ROUNDS} rounds  |  {POSE_HOLD_SECONDS}s per pose", h//2 + 40, size=0.7, color=YELLOW)
                draw_centered_text(frame, "Press SPACE to start", h//2 + 100, size=0.9, color=GREEN)

            # ══════════════════════════════════════════════════
            # STATE: COUNTDOWN
            # ══════════════════════════════════════════════════
            elif state == STATE_COUNTDOWN:
                show_reference_image(frame, current_pose["image"], current_pose["silhouette"])

                # Draw silhouette on main frame too
                draw_silhouette(frame, current_pose["silhouette"], color=(0, 200, 200), alpha=0.3)

                remaining = COUNTDOWN_SECONDS - elapsed
                count_text = str(int(remaining) + 1) if remaining > 0 else "GO!"
                count_color = RED if remaining > 1 else GREEN

                draw_centered_text(frame, f"Round {round_index + 1} of {TOTAL_ROUNDS}", 50, size=0.8, color=WHITE)
                draw_centered_text(frame, "GET READY!", h//2 - 60, size=1.2, color=YELLOW)
                draw_centered_text(frame, count_text, h//2 + 20, size=3.0, color=count_color)
                draw_centered_text(frame, current_pose["name"].replace("_", " ").upper(), h - 60, size=0.7, color=CYAN)

                if elapsed >= COUNTDOWN_SECONDS:
                    state = STATE_PLAYING
                    state_start = now
                    best_score = 0

            # ══════════════════════════════════════════════════
            # STATE: PLAYING
            # ══════════════════════════════════════════════════
            elif state == STATE_PLAYING:
                show_reference_image(frame, current_pose["image"], current_pose["silhouette"])
                draw_silhouette(frame, current_pose["silhouette"], color=(0, 180, 180), alpha=0.25)

                time_left = POSE_HOLD_SECONDS - elapsed

                if player_angles:
                    round_score = score_pose(current_pose["angles"], player_angles)
                    best_score = max(best_score, round_score)

                    # Score colour
                    if round_score >= WIN_THRESHOLD:
                        score_color = GREEN
                    elif round_score >= 50:
                        score_color = YELLOW
                    else:
                        score_color = RED

                    # Big score
                    draw_centered_text(frame, f"{round_score:.0f}%", h//2, size=3.5, color=score_color)

                    # Joint feedback (small, top-left)
                    y_off = 30
                    for joint, angle in player_angles.items():
                        target_angle = current_pose["angles"][joint]
                        diff = abs(angle - target_angle)
                        jcolor = GREEN if diff < 20 else YELLOW if diff < 40 else RED
                        cv2.putText(frame, f"{joint[:8]}: {int(angle)}° (target {int(target_angle)}°)",
                                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.38, jcolor, 1)
                        y_off += 16
                else:
                    draw_centered_text(frame, "NO PERSON DETECTED", h//2, size=1.0, color=RED)

                # Timer bar
                bar_color = GREEN if time_left > 2 else YELLOW if time_left > 1 else RED
                draw_timer_bar(frame, elapsed, POSE_HOLD_SECONDS, color=bar_color)
                draw_centered_text(frame, f"{max(0, time_left):.1f}s", h - 50, size=0.7, color=WHITE)
                draw_centered_text(frame, f"Round {round_index + 1}/{TOTAL_ROUNDS}", 35, size=0.7, color=WHITE)

                if elapsed >= POSE_HOLD_SECONDS:
                    scores_history.append(best_score)
                    total_score += best_score
                    if best_score >= WIN_THRESHOLD:
                        flash_timer = 8
                    state = STATE_RESULT
                    state_start = now

            # ══════════════════════════════════════════════════
            # STATE: RESULT
            # ══════════════════════════════════════════════════
            elif state == STATE_RESULT:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), BLACK, -1)
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

                passed = best_score >= WIN_THRESHOLD
                result_text = "NAILED IT! 🎉" if passed else "MISSED IT 😅"
                result_color = GREEN if passed else RED

                draw_centered_text(frame, result_text, h//2 - 60, size=1.3, color=result_color)
                draw_centered_text(frame, f"Best: {best_score:.0f}%", h//2, size=2.0, color=result_color)
                draw_centered_text(frame, f"Need {WIN_THRESHOLD}% to pass", h//2 + 60, size=0.7, color=WHITE)

                if elapsed < 2:
                    progress = f"Round {round_index + 1}/{TOTAL_ROUNDS} complete"
                    draw_centered_text(frame, progress, h - 60, size=0.7, color=YELLOW)
                else:
                    draw_centered_text(frame, "Next pose...", h - 60, size=0.7, color=CYAN)

                if elapsed >= 3:
                    round_index += 1
                    if round_index >= TOTAL_ROUNDS:
                        state = STATE_GAMEOVER
                    else:
                        current_pose = selected_poses[round_index]
                        state = STATE_COUNTDOWN
                    state_start = now

            # ══════════════════════════════════════════════════
            # STATE: GAME OVER
            # ══════════════════════════════════════════════════
            elif state == STATE_GAMEOVER:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), BLACK, -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                avg = total_score / TOTAL_ROUNDS
                passed_rounds = sum(1 for s in scores_history if s >= WIN_THRESHOLD)

                if avg >= 80:
                    title, color = "DANCE MASTER!", CYAN
                elif avg >= 60:
                    title, color = "GREAT JOB!", GREEN
                elif avg >= 40:
                    title, color = "KEEP PRACTICING!", YELLOW
                else:
                    title, color = "BETTER LUCK NEXT TIME!", RED

                draw_centered_text(frame, "GAME OVER", h//2 - 140, size=2.0, color=WHITE)
                draw_centered_text(frame, title, h//2 - 70, size=1.3, color=color)
                draw_centered_text(frame, f"Average Score: {avg:.1f}%", h//2, size=1.2, color=color)
                draw_centered_text(frame, f"Poses passed: {passed_rounds}/{TOTAL_ROUNDS}", h//2 + 60, size=0.9, color=WHITE)

                # Show per-round scores
                y = h//2 + 110
                for i, s in enumerate(scores_history):
                    sc = GREEN if s >= WIN_THRESHOLD else RED
                    cv2.putText(frame, f"  Round {i+1}: {s:.0f}%",
                                (w//2 - 80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)
                    y += 25

                draw_centered_text(frame, "SPACE = play again   Q = quit", h - 30, size=0.6, color=WHITE)

            # ── SHOW FRAME ────────────────────────────────────
            cv2.imshow("PoseMaster 🎮", frame)

            # ── KEY HANDLING ──────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord(' '):
                if state == STATE_INTRO:
                    selected_poses = random.sample(poses, min(TOTAL_ROUNDS, len(poses)))
                    current_pose = selected_poses[0]
                    round_index = 0
                    total_score = 0
                    scores_history = []
                    state = STATE_COUNTDOWN
                    state_start = time.time()

                elif state == STATE_GAMEOVER:
                    selected_poses = random.sample(poses, min(TOTAL_ROUNDS, len(poses)))
                    current_pose = selected_poses[0]
                    round_index = 0
                    total_score = 0
                    scores_history = []
                    best_score = 0
                    state = STATE_COUNTDOWN
                    state_start = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Thanks for playing PoseMaster!")

if __name__ == "__main__":
    run_game()