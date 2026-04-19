import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide TF logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import cv2
import pickle
from src.recognize import recognize_face
from datetime import datetime
from deepface import DeepFace
import numpy as  np

from src.utils import normalize

# Load embeddings
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

known_names = []
known_embeddings = []
for name, emb_list in data.items():
    for emb in emb_list:
        known_names.append(name)
        known_embeddings.append(emb)


known_embeddings = [normalize(np.array(e)) for e in known_embeddings]

# print("Data keys:", data.keys())

# for name, emb_list in data.items():
#     print(name, "->", len(emb_list))


def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")

    with open("attendance.csv", "a+", newline="") as f:
        f.seek(0)
        lines = f.readlines()

        entries = [line.strip().split(",") for line in lines if line.strip()]
        for e in entries:
            if name==e[0]   and today==e[1]:
               return

        time = datetime.now().strftime("%H:%M:%S")
        f.write(f"{name},{today},{time}\n")

cap = cv2.VideoCapture(0)
faces = []
names = []
frame_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1

    if not ret:
        break

    # Update detection + recognition every 5 frames
    if frame_count % 5 == 0:
        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        new_faces = DeepFace.extract_faces(small_frame, enforce_detection=False)

        if new_faces:   # only update if faces found
            faces = new_faces
            names = []

            for face in faces:
                area = face["facial_area"]
                x, y, w, h = area["x"], area["y"], area["w"], area["h"]
                x, y, w, h = int(x*2), int(y*2), int(w*2), int(h*2)

                face_img = frame[y:y+h, x:x+w]
                name = recognize_face(face_img, known_embeddings, known_names)

                names.append(name)

                if name != "Unknown":
                    mark_attendance(name)

    # Draw using last known data (IMPORTANT)
    for i, face in enumerate(faces):
        area = face["facial_area"]
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]
        x, y, w, h = int(x*2), int(y*2), int(w*2), int(h*2)
        x1, y1, x2, y2 = x, y, x+w, y+h

        name = names[i] if i < len(names) else "Unknown"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(frame, name, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()