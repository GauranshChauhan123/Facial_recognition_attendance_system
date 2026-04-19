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


for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path,person)
    if  not os.path.isdir(person_path):
        continue
    
    if person not in data:
        data[person]=[]
    for file in os.listdir(person_path): 
            if not file.lower().endswith(('.jpeg','.jpg','.png')): 
                continue 
            else:

                image_path=os.path.join(person_path,file)
                try:
                    
                    encodings = DeepFace.represent(
                         img_path=image_path,
                         model_name='ArcFace',
                         enforce_detection=False,
                         detector_backend="opencv"
                        )
    
                    if len(encodings) == 0:
                        continue
                    
                    embedding = encodings[0]["embedding"]
                    if embedding not in data[person]:
                         data[person].append(embedding)
                         print(f"✅ Added: {file}")


                except Exception as e:
                      print(f"error {e}")


with open(file_path, "wb") as f:
         pickle.dump(data, f)

print("Saved successfully!")







    
    