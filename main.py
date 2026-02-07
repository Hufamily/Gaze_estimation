import mediapipe as mp
import cv2
import gaze
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions

# Path to the face landmarker model
model_path = "face_landmarker.task"

# Create a face landmarker instance:
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path))

# camera stream:
cap = cv2.VideoCapture(0)  # camera index 0

with FaceLandmarker.create_from_options(options) as face_landmarker:
    while cap.isOpened():
        success, image = cap.read()
        if not success:  # no frame input
            print("Ignoring empty camera frame.")
            continue
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # Detect face landmarks
        results = face_landmarker.detect(mp_image)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV

        if results.face_landmarks:
            gaze.gaze(image, results.face_landmarks[0])  # gaze estimation

        cv2.imshow('output window', image)
        if cv2.waitKey(2) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
