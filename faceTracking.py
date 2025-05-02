import cv2
import numpy as np
import time
import math

# Load OpenCV DNN face detector
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize strike count and timers
strike_count = 0
last_strike_time = 0
cooldown_time = 6  # Avoid rapid strikes
start_time = time.time()
calm_start_duration = 5  # Wait time before strikes are allowed

# Dead zone thresholds
DEAD_ZONE_YAW = 80
DEAD_ZONE_PITCH = 50

# Duration pose must persist to count as strike
POSE_PERSISTENCE_DURATION = 2
pose_start_time = None
pose_active = False

def auto_brightness(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mean_v = np.mean(v)
    if mean_v < 100:
        factor = 130 / mean_v
        hsv[:, :, 2] = np.clip(v * factor, 0, 255)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif mean_v > 180:
        factor = 180 / mean_v
        hsv[:, :, 2] = np.clip(v * factor, 0, 255)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame

model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype="double")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = auto_brightness(frame)
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    suspicious_pose = False
    pitch = yaw = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX, endY))
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = [(int((startX + endX) / 2), int((startY + endY) / 2)),
                         (int((startX + endX) / 2), endY),
                         (startX, startY),
                         (endX, startY),
                         (startX, endY),
                         (endX, endY)]

            image_points = np.array(landmarks, dtype="double")
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))
            success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
            rmat, _ = cv2.Rodrigues(rotation_vector)
            proj_matrix = np.hstack((rmat, translation_vector))
            euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

            pitch, yaw = float(euler_angles[0]), float(euler_angles[1])

            cv2.putText(frame, f"Pitch: {int(pitch)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {int(yaw)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Updated suspicious pose logic
            if abs(yaw) > DEAD_ZONE_YAW or abs(pitch) > DEAD_ZONE_PITCH:
                if not pose_active:
                    pose_start_time = time.time()
                    pose_active = True
                elif time.time() - pose_start_time >= POSE_PERSISTENCE_DURATION:
                    suspicious_pose = True
            else:
                if pose_active and (time.time() - pose_start_time < POSE_PERSISTENCE_DURATION):
                    pass  # Wait for full duration before resetting
                else:
                    pose_active = False
                    pose_start_time = None

    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time > calm_start_duration and current_time - last_strike_time > cooldown_time:
        if len(faces) == 0:
            strike_count += 1
            last_strike_time = current_time
            print("No face detected - Strike", strike_count)
        elif len(faces) > 1:
            strike_count += 1
            last_strike_time = current_time
            print("Multiple faces detected - Strike", strike_count)
        elif suspicious_pose:
            strike_count += 1
            last_strike_time = current_time
            print("Suspicious head pose - Strike", strike_count)

    cv2.putText(frame, f"Strikes: {strike_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("DNN Proctoring System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or strike_count >= 3:
        break

cap.release()
cv2.destroyAllWindows()
