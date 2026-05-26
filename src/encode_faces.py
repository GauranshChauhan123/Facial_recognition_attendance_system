import cv2
import numpy as np
import os
import pickle

# =========================
# LOAD MODELS
# =========================

detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320)
)

recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx",
    ""
)

# =========================
# DATASET PATH
# =========================

dataset_path = "data/known_faces"

# =========================
# STORAGE
# =========================

embeddings = []
names = []

# =========================
# PROCESS DATASET
# =========================

for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    person_embeddings = []

    print(f"\nProcessing {person_name}...")

    for image_name in os.listdir(person_folder):

        img_path = os.path.join(person_folder, image_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w, _ = img.shape

        # Set detector input size
        detector.setInputSize((w, h))

        # Detect face
        _, faces = detector.detect(img)

        if faces is None:
            print(f" No face found in {image_name}")
            continue

        # Take first detected face
        face = faces[0]

        try:
            # Align face properly
            aligned_face = recognizer.alignCrop(img, face)

            # Generate embedding
            feature = recognizer.feature(aligned_face)

            # Flatten to (128,)
            feature = feature.flatten()

            person_embeddings.append(feature)

            print(f" Processed: {image_name}")

        except Exception as e:
            print(f" Error in {image_name}: {e}")

    # =========================
    # AVERAGE EMBEDDINGS
    # =========================

    if len(person_embeddings) > 0:

        avg_embedding = np.mean(person_embeddings, axis=0)

        embeddings.append(avg_embedding)
        names.append(person_name)

        print(f" Saved embedding for {person_name}")

# =========================
# SAVE FILES
# =========================

embeddings = np.array(embeddings)

np.save("embeddings.npy", embeddings)

with open("names.pkl", "wb") as f:
    pickle.dump(names, f)

print("\n================================")
print("Training Complete")
print(f"Total Persons Enrolled: {len(names)}")
print("Embeddings saved to embeddings.npy")
print("Names saved to names.pkl")
print("================================")