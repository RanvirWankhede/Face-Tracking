import cv2
import dlib
import numpy as np
import time

# Initialize face detector and shape predictor (HOG-based)
hog_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Strike & accuracy tracking
strike_count = 0
last_strike_time = 0
total_frames = 0
face_detected_frames = 0
exam_terminated = False

# Thresholds and constants
YAW_THRESHOLD = 180
PITCH_THRESHOLD = 10
DEAD_ZONE = 12
STRIKE_DELAY = 6

model_points = np.array([
    (0.0, 0.0, 0.0),               # Nose tip
    (0.0, -330.0, -65.0),          # Chin
    (-225.0, 170.0, -135.0),       # Left eye
    (225.0, 170.0, -135.0),        # Right eye
    (-150.0, -150.0, -125.0),      # Left mouth
    (150.0, -150.0, -125.0)        # Right mouth
], dtype="double")

# Camera calibration matrix
focal_length = 640
center = (320, 240)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype="double")

dist_coeffs = np.zeros((4, 1))  # No lens distortion

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    total_frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_detector(gray)

    # If no face is detected, show warning
    if len(faces) == 0:
        cv2.putText(frame, "NO FACE DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    # If more than one face is detected, treat it as a violation too
    elif len(faces) > 1:
        cv2.putText(frame, "MULTIPLE FACES DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        # Mark violation (no head pose in this case)
        if time.time() - last_strike_time > STRIKE_DELAY:
            strike_count += 1
            last_strike_time = time.time()
            print(f"STRIKE {strike_count}/3 - Multiple Faces Detected!")
    else:
        face_detected_frames += 1
        # Process the first detected face
        shape = predictor(gray, faces[0])
        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),  # Nose tip
            (shape.part(8).x, shape.part(8).y),    # Chin
            (shape.part(36).x, shape.part(36).y),  # Left eye
            (shape.part(45).x, shape.part(45).y),  # Right eye
            (shape.part(48).x, shape.part(48).y),  # Left mouth
            (shape.part(54).x, shape.part(54).y)   # Right mouth
        ], dtype="double")
        success, rotation_vector, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)

        if success:
            rmat, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            yaw, pitch, roll = int(angles[0]), int(angles[1]), int(angles[2])

            # Display angles for debugging
            cv2.putText(frame, f"Yaw: {yaw}  Pitch: {pitch}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Ignore small movements (dead zone)
            if abs(yaw) <= DEAD_ZONE and abs(pitch) <= DEAD_ZONE:
                pass
            # Check if head pose violation occurs
            elif abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD:
                if time.time() - last_strike_time > STRIKE_DELAY:
                    strike_count += 1
                    last_strike_time = time.time()
                    print(f"STRIKE {strike_count}/3 - Head Pose Violation!")
                    cv2.putText(frame, f"STRIKE {strike_count}/3", (50, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Check if exam should be terminated
    if strike_count >= 3 and not exam_terminated:
        print("Exam Terminated Due to Multiple Violations!")
        exam_terminated = True
        cv2.putText(frame, "EXAM TERMINATED", (100, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv2.imshow("Live Proctoring", frame)
        cv2.waitKey(3000)
        break  # Break out of the loop gracefully

    cv2.imshow("Live Proctoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop, print accuracy stats
accuracy = (face_detected_frames / total_frames) * 100 if total_frames > 0 else 0
print(f"\nCamera Accuracy: {accuracy:.2f}%")
print(f"Total Frames Processed: {total_frames}")
print(f"Frames with Face Detected: {face_detected_frames}")

# Cleanup resources
cap.release()
cv2.destroyAllWindows()
