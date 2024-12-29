import cv2
import dlib
import numpy as np
from datetime import datetime
import time
import os


class FaceTracking:
    def __init__(self):
        print("Initializing FaceTracking...")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"

        # Check if predictor file exists
        if not os.path.exists(self.predictor_path):
            raise FileNotFoundError(f"Predictor file '{self.predictor_path}' not found.")

        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.detector = dlib.get_frontal_face_detector()
        self.warning_count = 0
        self.warning_limit = 3
        self.baseline_horizontal_angle = None
        self.baseline_vertical_position = None
        self.warmup_frames = 50
        self.frame_counter = 0
        self.is_warmup_complete = False
        print("FaceTracking initialized successfully.")

    def log_timestamp(self, message):
        """Logs a message with a timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def reset_baselines(self):
        """Resets baseline angles and positions."""
        self.baseline_horizontal_angle = None
        self.baseline_vertical_position = None
        print("Baselines reset.")

    def start_tracking(self):
        """Starts the face tracking process."""
        print("Starting face tracking...")
        cap = cv2.VideoCapture(0)

        # Check if the camera opened successfully
        if not cap.isOpened():
            raise RuntimeError("Error: Could not access the webcam. Make sure it's connected and available.")

        self.log_timestamp("Face tracking started.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame from webcam.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)

                # Warm-up phase
                if self.frame_counter < self.warmup_frames:
                    cv2.putText(frame, "Adjust your position...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    self.frame_counter += 1
                    cv2.imshow('Face Tracking with Angle', frame)
                    if self.frame_counter == self.warmup_frames:
                        self.is_warmup_complete = True
                        self.reset_baselines()
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # Check for multiple faces
                if len(faces) > 1:
                    self.warning_count += 1
                    warning_text = f"Warning {self.warning_count}/{self.warning_limit}: Multiple faces detected!"
                    cv2.putText(frame, warning_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if self.warning_count >= self.warning_limit:
                        self.log_timestamp("Multiple faces detected too many times. Exiting.")
                        break
                    cv2.imshow('Face Tracking with Angle', frame)
                    cv2.waitKey(1)
                    time.sleep(5)
                    continue

                # Detected face process
                for face in faces:
                    landmarks = self.predictor(gray, face)
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Eye and nose tip landmarks
                    left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
                    right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])
                    nose_tip = np.array([landmarks.part(30).x, landmarks.part(30).y])

                    # Angle and deviation
                    delta_x = right_eye[0] - left_eye[0]
                    delta_y = right_eye[1] - left_eye[1]
                    current_horizontal_angle = np.degrees(np.arctan2(delta_y, delta_x))
                    current_vertical_position = nose_tip[1]

                    if self.baseline_horizontal_angle is None:
                        self.baseline_horizontal_angle = current_horizontal_angle
                    if self.baseline_vertical_position is None:
                        self.baseline_vertical_position = current_vertical_position

                    horizontal_deviation = abs(current_horizontal_angle - self.baseline_horizontal_angle)
                    vertical_deviation = abs(current_vertical_position - self.baseline_vertical_position)

                    # Display tracking data
                    angle_text = f"Angle: {current_horizontal_angle:.2f} degrees"
                    deviation_text = f"Left-Right Deviation: {horizontal_deviation:.2f}, Up-Down Deviation: {vertical_deviation:.2f}"
                    cv2.putText(frame, angle_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, deviation_text, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Check for excessive movement
                    if horizontal_deviation > 1.5 or vertical_deviation > 60:
                        self.warning_count += 1
                        warning_text = f"Warning {self.warning_count}/{self.warning_limit}: Excessive head movement detected!"
                        cv2.putText(frame, warning_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if self.warning_count >= self.warning_limit:
                            self.log_timestamp("Excessive head movement too many times. Exiting.")
                            break
                        time.sleep(5)
                        continue
                if self.warning_count >= self.warning_limit:
                    self.log_timestamp("Excessive head movement too many times. Exiting.")
                    break

                cv2.imshow('Face Tracking with Angle', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.log_timestamp("Face tracking ended.")


if __name__ == "__main__":
    try:
        tracker = FaceTracking()
        tracker.start_tracking()
    except Exception as e:
        print(f"Failed to start face tracking: {e}")
