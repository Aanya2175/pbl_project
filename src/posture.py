import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# global state
pose = None
baseline = None
stable_frames = 0
calibration_frames = 0


def detect_posture(frame):
    global pose, baseline, stable_frames, calibration_frames

    # -------- initialise mediapipe lazily (Cloud-safe) --------
    if pose is None:
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    posture_status = "Unknown"
    slouch_value = 0

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        nose = lm[mp_pose.PoseLandmark.NOSE.value]

        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        # forward slouch
        slouch_value = nose.y - shoulder_y

        # sideways tilt
        tilt = abs(left_shoulder.y - right_shoulder.y)

        # -------- calibration --------
        if baseline is None:
            calibration_frames += 1
            posture_status = "Calibrating"

            if calibration_frames > 60:
                baseline = slouch_value

            return posture_status, round(slouch_value, 3)

        diff = slouch_value - baseline

        # -------- posture logic --------
        if diff > 0.05:
            posture_status = "Bad"
            stable_frames = 0

        elif tilt > 0.05:
            posture_status = "Bad"
            stable_frames = 0

        else:
            posture_status = "Good"
            stable_frames += 1

        # -------- auto recalibration --------
        if stable_frames > 120:
            baseline = slouch_value
            stable_frames = 0

    return posture_status, round(slouch_value, 3)
