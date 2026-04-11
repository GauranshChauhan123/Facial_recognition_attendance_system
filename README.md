# 🎯 Face Recognition Attendance System

A real-time face recognition-based attendance system built using **DeepFace**, **OpenCV**, and **Python**.
The system detects faces from a webcam, recognizes individuals using stored embeddings, and automatically marks attendance.

---

## 🚀 Features

* 🎥 Real-time face detection using webcam
* 🧠 Face recognition using DeepFace embeddings
* 📊 Automatic attendance marking (CSV file)
* 🗓️ Duplicate prevention (one entry per day)
* 👥 Supports multiple faces
* ⚡ Optimized for performance (frame skipping + caching)

---

## 🛠️ Tech Stack

* Python
* OpenCV
* DeepFace
* TensorFlow
* NumPy

---

## 📂 Project Structure

```
Attendance_System/
│── src/
│   |── recognize.py        # Face recognition logic
│   | ── encodes.py 
│── main.py                 # Main execution file
│                           # Generate embeddings
│── embeddings.pkl          # Stored face embeddings (ignored)
│── attendance.csv          # Attendance records (auto-generated)
│
│── data/                   # Add images here (ignored)
│── .gitignore
│── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/GauranshChauhan123/Facial_recognition_attendance_system
cd attendance-system
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## 📸 How to Use

### Step 1: Add images

* Create a folder named `data/`
* Add images of persons (one person per image)

---

### Step 2: Generate embeddings

```
python encodes.py
```

👉 This creates `embeddings.pkl`

---

### Step 3: Run the system

```
python main.py
```

---

### Step 4: Attendance output

* Stored in `attendance.csv`
* Format:

```
Name,Date,Time
abhi,2026-04-11,10:30:22
```

---

## ⚡ Performance Optimizations

* Frame skipping (runs detection every few frames)
* Cached face detection to avoid flickering
* Per-face recognition instead of full-frame processing

---

## ⚠️ Notes

* `data/`, `attendance.csv`, and `embeddings.pkl` are ignored in Git
* Ensure good lighting for better accuracy
* Tune threshold for better recognition results

---

## 🔮 Future Improvements

* GUI (Tkinter / Streamlit)
* Database integration (SQLite/MySQL)
* Face tracking for smoother performance
* Export to Excel
* Live dashboard for attendance

---

## 👨‍💻 Author

GAURANSH CHAUHAN

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
