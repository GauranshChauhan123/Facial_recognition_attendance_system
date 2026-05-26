import cv2
import pandas as pd
import os

from src.recognize import recognize_face
from src.utils import draw_box, mark_attendance

# LOAD MODELS

detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (640, 640)
)

recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx",
    ""
)

# ATTENDANCE FILE

attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):

    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(attendance_file, index=False)

attendance_marked = set()



cap = cv2.VideoCapture(0)

print("Face Recognition Attendance Started")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            try:

                
                aligned_face = recognizer.alignCrop(frame, face)

                # GENERATE EMBEDDING

                feature = recognizer.feature(aligned_face)

                feature = feature.flatten()

                
                # RECOGNIZE FACE

                name, score = recognize_face(feature)

                
                draw_box(frame, face, name, score)

                
                # MARK ATTENDANCE
                

                if name != "Unknown":

                    mark_attendance(
                        name,
                        attendance_file
                    )

            except Exception as e:

                print("Error:", e)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

cv2.destroyAllWindows()