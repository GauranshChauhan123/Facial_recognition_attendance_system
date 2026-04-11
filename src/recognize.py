
from deepface import DeepFace
import numpy as np

def recognize_face(frame, known_embeddings, known_names):
    try:
        # Get embedding of current frame face
        result = DeepFace.represent(frame, model_name="ArcFace", enforce_detection=False)

        if len(result) == 0:
            return "Unknown"

        embedding = result[0]["embedding"]

        # Compare with stored embeddings
        distances = [np.linalg.norm(np.array(embedding) - np.array(k)) for k in known_embeddings]
        min_dist = min(distances)

        threshold = 10

        if min_dist < threshold:
            index = distances.index(min_dist)
            return known_names[index]
        else:
            return "Unknown"

    except:
        return "Unknown"