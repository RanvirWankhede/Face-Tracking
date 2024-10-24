import cv2
from gaze_tracking import GazeTracking

#Initialize the GazeTracking object and access the webcam
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

#Check if the webcam is opened successfully
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        #Capture frame from the webcam
        ret, frame = webcam.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        #Analyze the frame using GazeTracking
        gaze.refresh(frame)

        #Get the annotated frame with gaze info
        frame = gaze.annotated_frame()

        #Determine the gaze direction
        text = ""
        if gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        #gaze direction
        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        #Show the frame with annotations
        cv2.imshow("Eyeball Detection", frame)

        #Exit
        if cv2.waitKey(1) == 27:
            break

except KeyboardInterrupt:
    #Handle interruption
    print("Process interrupted by user.")

finally:
    #Release webcam and close windows
    webcam.release()
    cv2.destroyAllWindows()
    print("Webcam released and all windows closed.")
