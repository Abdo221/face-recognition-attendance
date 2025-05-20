# Face Recognition Attendance System

This is a simple Python-based face recognition system using OpenCV and the `face_recognition` library. It identifies faces from webcam feed and logs recognized individuals with timestamps.

## 🚀 Features

- Real-time face detection and recognition
- Logs recognized names with time in a CSV file
- Easy to extend with more known faces
- Built with OpenCV + face_recognition

## 📁 Project Structure

face-recognition-attendance/
├── known/ # Folder with known face images (e.g., john.jpg)
├── recognition_log.csv # Logs of recognized names
├── face_recognition_script.py
├── README.md
└── .gitignore

markdown
Copy
Edit

## 🔧 Setup

1. Install dependencies:
   ```bash
   pip install face_recognition opencv-python numpy
Add known faces in the known/ folder. Each image should be named as the person's name (e.g., Abdo.jpg).

Run the app:

bash
Copy
Edit
python face_recognition_script.py