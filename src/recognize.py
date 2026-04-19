from deepface import DeepFace
import numpy as np

from src.utils import cosine_similarity,normalize

def recognize_face(frame, known_embeddings, known_names):
    try:
        result = DeepFace.represent(frame, model_name="ArcFace", enforce_detection=True)

        if len(result) == 0:
            return "Unknown"

        embedding = normalize(np.array(result[0]["embedding"]))

        similarities = []
        for k in known_embeddings:
            sim = cosine_similarity(embedding, k)
            similarities.append(sim)            

        max_sim = max(similarities)
        index = similarities.index(max_sim)

        print(f"Max sim: {max_sim:.3f}, Match: {known_names[index]}")

        threshold = 0.45

        if max_sim > threshold:
            return known_names[index]
        else:
            return "Unknown"

    except Exception as e:
        print("Error:", e)
        return "Unknown"