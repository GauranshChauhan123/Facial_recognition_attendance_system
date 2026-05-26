# Face Recognition Attendance System

A real-time Face Recognition Attendance System built using OpenCV, YuNet, and SFace.

This project detects faces from a webcam, recognizes registered users using facial embeddings, and automatically marks attendance with date and time in a CSV file.

---

# Features

- Real-time face recognition
- Automatic attendance marking
- Date and time logging
- Duplicate attendance prevention
- Unknown face detection
- Multiple images per person support
- Fast and lightweight system
- Modular project architecture

---

# Technologies Used

- Python
- OpenCV
- YuNet Face Detector
- SFace Face Recognizer
- NumPy
- Pandas

---

# Project Structure
```bash
project/
│
├── dataset/
│    
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
│
├── embeddings.npy
├── names.pkl
├── attendance.csv
│
├── encode_faces.py
├── recognize.py
├── utils.py
├── main.py
│
├── requirements.txt
└── README.md
```

---

# System Workflow

```text
Webcam Frame
      ↓
YuNet Face Detection
      ↓
Face Alignment
      ↓
SFace Embedding Extraction
      ↓
Cosine Similarity Matching
      ↓
Recognized / Unknown
      ↓
Attendance Marked
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/GauranshChauhan123/Facial_recognition_attendance_system
cd project-folder
```

---

## 2. Create Virtual Environment 

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# requirements.txt

```txt
opencv-contrib-python==4.10.0.84
numpy==1.26.4
pandas==2.2.2
```

---

# Download Required Models

Create a folder named:

```bash
models/
```

Download these ONNX models from OpenCV Zoo:

## YuNet Face Detector

```text
face_detection_yunet_2023mar.onnx
```

## SFace Face Recognizer

```text
face_recognition_sface_2021dec.onnx
```

Download Link:

https://github.com/opencv/opencv_zoo

Place both files inside the `models/` directory.

---

# Dataset Preparation

Inside the `dataset/` folder:

- Create one folder per person
- Add multiple images for each person

Example:

```bash
dataset/
│
├── Abhi/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
│
├── Rahul/
│   ├── 1.jpg
│   └── 2.jpg
```

## Recommendations

- Use clear front-face images
- Add 5–10 images per person
- Use different lighting conditions and angles
- Avoid blurry images

---

# Generate Face Embeddings

Run the following command:

```bash
python encode_faces.py
```

This script will:

- Detect faces from dataset images
- Generate facial embeddings using SFace
- Store embeddings in `embeddings.npy`
- Store names in `names.pkl`

---

# Start Face Recognition Attendance System

Run:

```bash
python main.py
```

The webcam will open and start real-time face recognition.

Press:

```text
ESC
```

to exit the application.

---

# Attendance System

Attendance is stored in:

```text
attendance.csv
```

Example:

```csv
Name,Date,Time
Abhi,2026-05-26,14:35:10
Rahul,2026-05-26,14:40:22
```

---

# Duplicate Attendance Prevention

The system prevents duplicate attendance entries for the same person on the same day.

Example:

- If Abhi is already marked today
- The system will not mark attendance again


# File Descriptions

## encode_faces.py

- Reads dataset images
- Detects faces
- Generates embeddings
- Saves embeddings and labels

---

## recognize.py

- Loads stored embeddings
- Compares embeddings using cosine similarity
- Returns matched name or "Unknown"

---

## utils.py

Contains utility/helper functions:

- Cosine similarity
- Drawing face boxes
- Attendance marking

---

## main.py

Main application file:

- Opens webcam
- Detects faces
- Generates embeddings
- Calls recognition module
- Marks attendance
- Displays results

---

# Future Improvements

- GUI Interface
- Streamlit Web App
- Database Integration
- Face Registration UI
- Anti-Spoofing
- Multi-Camera Support
- Cloud Attendance System
- Email Notifications

---

# License

This project is open-source and available for educational purposes.