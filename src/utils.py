import numpy as np


def normalize(v):
    return v / np.linalg.norm(v)

def cosine_similarity(a, b):
    return np.dot(a, b)