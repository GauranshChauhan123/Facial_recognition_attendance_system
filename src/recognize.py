import numpy as np
import pickle

from src.utils import cosine_similarity



embeddings = np.load("embeddings.npy")

with open("names.pkl", "rb") as f:
    names = pickle.load(f)



THRESHOLD = 0.45


def recognize_face(face_embedding):

    best_score = -1
    best_match = "Unknown"

    for i, stored_embedding in enumerate(embeddings):

        score = cosine_similarity(
            stored_embedding,
            face_embedding
        )

        if score > best_score:

            best_score = score
            best_match = names[i]

    if best_score < THRESHOLD:

        return "Unknown", best_score

    return best_match, best_score