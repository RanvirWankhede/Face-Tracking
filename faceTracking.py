import cv2
import dlib
import numpy as np
import time

def face_tracking():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)
    warning_count = 0
    baseline_horizontal_angle = None
    baseline_vertical_position = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 1:
            warning_count += 1
            warning_text = f"Warning {warning_count}/3: Multiple faces detected!"
            cv2.putText(frame, warning_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Face Tracking with Angle', frame)
            cv2.waitKey(1)
            if warning_count >= 3:
                print("Multiple faces detected or excessive head movement multiple times. Exiting program.")
                break
            time.sleep(5)
            continue

        for face in faces:
            landmarks = predictor(gray, face)
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
            right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])
            nose_tip = np.array([landmarks.part(30).x, landmarks.part(30).y])
            delta_x = right_eye[0] - left_eye[0]
            delta_y = right_eye[1] - left_eye[1]
            current_horizontal_angle = np.degrees(np.arctan2(delta_y, delta_x))
            current_vertical_position = nose_tip[1]

            if baseline_horizontal_angle is None:
                baseline_horizontal_angle = current_horizontal_angle
            if baseline_vertical_position is None:
                baseline_vertical_position = current_vertical_position

            horizontal_deviation = abs(current_horizontal_angle - baseline_horizontal_angle)
            vertical_deviation = abs(current_vertical_position - baseline_vertical_position)
            angle_text = f"Angle: {current_horizontal_angle:.2f} degrees"
            deviation_text = f"Left-Right Deviation: {horizontal_deviation:.2f}, Up-Down Deviation: {vertical_deviation:.2f}"
            cv2.putText(frame, angle_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, deviation_text, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if horizontal_deviation > 1.5 or vertical_deviation > 60:
                warning_count += 1
                warning_text = f"Warning {warning_count}/3: Excessive head movement detected!"
                cv2.putText(frame, warning_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Face Tracking with Angle', frame)
                cv2.waitKey(1)
                if warning_count >= 3:
                    print("Multiple faces detected or excessive head movement multiple times. Exiting program.")
                    break
                time.sleep(5)
                continue

        if warning_count > 3:
            print("Multiple faces detected or excessive head movement multiple times. Exiting program.")
            break

        cv2.imshow('Face Tracking with Angle', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

face_tracking()
