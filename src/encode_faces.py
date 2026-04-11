import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional: remove oneDNN warning

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import pickle
from deepface import DeepFace

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "embeddings.pkl")
dataset_path= os.path.join(BASE_DIR, "data","known_faces")

if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                   with open(file_path, "rb") as f:
                       data = pickle.load(f)

else:
       data={}


for file in os.listdir(dataset_path):
    if  not file.lower().endswith(('.jpeg','.jpg','.png')):
        continue
    else:
        image_path=os.path.join(dataset_path,file)
        name=os.path.splitext(file)[0]
        if name in data:
               continue
        encodings = DeepFace.represent(
             img_path=image_path,
             model_name='ArcFace',
             enforce_detection=False,
             detector_backend="opencv"
           )
    
        if len(encodings) == 0:
            continue
        else:
             embedding = encodings[0]["embedding"]
             data[name] = embedding

     
with open(file_path, "wb") as f:
         pickle.dump(data, f)

print("Saved successfully!")







    
    