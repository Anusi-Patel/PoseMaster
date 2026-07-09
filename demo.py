import cv2
import mediapipe as mp
import numpy as np

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ── ANGLE MATH ────────────────────────────────────────────────
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

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

# ── SCORING ENGINE ────────────────────────────────────────────
def score_pose(target_angles, player_angles, threshold=30):
    """
    Compare target pose vs player pose.
    Each joint: full 100pts if perfect, 0pts if off by threshold degrees.
    Returns overall score 0-100.
    """
    scores = []
    for joint in target_angles:
        diff = abs(target_angles[joint] - player_angles[joint])
        joint_score = max(0, 100 - (diff / threshold * 100))
        scores.append(joint_score)
    return round(sum(scores) / len(scores), 1)

# ── STATE ─────────────────────────────────────────────────────
target_angles = None   # saved when you press S
current_score = None   # live score vs target

# ── WEBCAM ────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("Controls:  S = save current pose as target  |  R = reset target  |  Q = quit")

with mp_pose.Pose() as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Draw skeleton
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Extract current angles
            current_angles = extract_key_angles(landmarks)

            # ── If target is saved, calculate and show score ──
            if target_angles is not None:
                current_score = score_pose(target_angles, current_angles)

                # Pick colour based on score
                if current_score >= 80:
                    score_color = (0, 255, 0)    # green  — nailed it
                elif current_score >= 50:
                    score_color = (0, 200, 255)  # yellow — getting there
                else:
                    score_color = (0, 0, 255)    # red    — way off

                # Big score in centre of screen
                cv2.putText(frame, f"{current_score}%",
                            (220, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, score_color, 8)

                # Label
                cv2.putText(frame, "Match target pose!",
                            (150, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ── Joint angles (top left) ──
            y_offset = 20
            for joint, angle in current_angles.items():
                cv2.putText(frame, f"{joint}: {int(angle)}",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y_offset += 18

        # ── Instructions (bottom of screen) ──
        if target_angles is None:
            instruction = "Press S to save your current pose as target"
            color = (200, 200, 200)
        else:
            instruction = "Press R to reset target pose"
            color = (100, 255, 100)

        cv2.putText(frame, instruction, (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imshow('PoseMaster - Stage 2', frame)

        # ── Key controls ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s') and results.pose_landmarks:
            # Save current pose as target
            target_angles = extract_key_angles(results.pose_landmarks.landmark)
            print("✅ Target pose saved:", {k: round(v, 1) for k, v in target_angles.items()})

        elif key == ord('r'):
            # Reset
            target_angles = None
            current_score = None
            print("🔄 Target reset.")

cap.release()
cv2.destroyAllWindows()