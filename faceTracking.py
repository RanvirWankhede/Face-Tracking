import cv2
import dlib
import numpy as np
import time

# Load CNN-based face detector
cnn_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Strike tracking
strike_count = 0
last_strike_time = 0  

# Accuracy tracking
total_frames = 0
face_detected_frames = 0

# **Thresholds (Updated)**
YAW_THRESHOLD = 200 
PITCH_THRESHOLD = 10  
DEAD_ZONE = 8  
STRIKE_DELAY = 5  

# 3D Model Points for Head Pose Estimation
model_points = np.array([
    (0.0, 0.0, 0.0),  
    (0.0, -330.0, -65.0),  
    (-225.0, 170.0, -135.0),  
    (225.0, 170.0, -135.0),  
    (-150.0, -150.0, -125.0),  
    (150.0, -150.0, -125.0)  
], dtype="double")

# Camera Calibration
focal_length = 640
center = (320, 240)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype="double")

dist_coeffs = np.zeros((4, 1))  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    total_frames += 1  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cnn_detector(gray, 1)  

    if not faces:  
        print("ERROR: No face detected!")
        cv2.putText(frame, "NO FACE DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        cv2.imshow("Live Proctoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue  

    face_detected_frames += 1  

    for face in faces:
        face_rect = face.rect  
        shape = predictor(gray, face_rect)

        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),  
            (shape.part(8).x, shape.part(8).y),  
            (shape.part(36).x, shape.part(36).y),  
            (shape.part(45).x, shape.part(45).y),  
            (shape.part(48).x, shape.part(48).y),  
            (shape.part(54).x, shape.part(54).y)  
        ], dtype="double")

        # SolvePnP to get head pose
        success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

        if success:
            rmat, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            yaw, pitch, roll = angles  

            # Debugging: Print angles to terminal
            print(f"Yaw: {int(yaw)}, Pitch: {int(pitch)}")  

            # Ignore small movements (Dead Zone)
            if abs(yaw) < DEAD_ZONE and abs(pitch) < DEAD_ZONE:
                continue  

            # **Strike System**
            if abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD:  
                if time.time() - last_strike_time > STRIKE_DELAY:  
                    strike_count += 1
                    last_strike_time = time.time()
                    print(f"STRIKE {strike_count}/3 (Looking Away Detected!)")

                cv2.putText(frame, f"STRIKE {strike_count}/3", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                if strike_count >= 3:
                    print("Exam Terminated Due to Multiple Warnings!")
                    cap.release()
                    cv2.destroyAllWindows()
                    break  

    cv2.imshow("Live Proctoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# **Calculate Accuracy**
accuracy = (face_detected_frames / total_frames) * 100 if total_frames > 0 else 0
print(f"\nCamera Accuracy: {accuracy:.2f}%")
print(f"Total Frames Processed: {total_frames}")
print(f"Frames with Face Detected: {face_detected_frames}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
