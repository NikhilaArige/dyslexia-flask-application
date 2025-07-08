# 🧠 Dyslexia Detection using AI (Flask App)

This project is a web-based dyslexia assessment tool that uses AI to analyze a user's **eye movements**, **handwriting**, and **speech patterns** to detect potential indicators of dyslexia. It is built using Python (Flask) and machine learning models trained with image and audio data.

---

## 📌 Features

- 👁️ Eye Tracking Module  
  Uses a trained CNN to analyze eye movement patterns.

- ✍️ Handwriting Module  
  Processes handwriting images to find common dyslexic writing traits.

- 🎤 Audio Module  
  Extracts MFCCs, ZCR, RMS, and spectral features to analyze speech rhythm and clarity.

- 📊 Results Page  
  Provides a final score and recommendation based on all three inputs.

---

## 📁 Folder Structure

project-root/
├── app.py # Main Flask application
├── dyslexia_audio_processing.py # Audio model (training + inference)
├── Eye_movement_training.py # Eye model training script
├── models/
│ ├── eye_movement_trained.h5
│ ├── audio_model.pkl
│ ├── audio_scaler.pkl
│ └── handwriting_model.h5 OR handwriting_dyslexia_model.pkl
├── templates/
│ ├── index.html
│ ├── eye_tracking.html
│ ├── handwriting.html
│ ├── audio.html
│ ├── results.html
│ ├── about.html
│ └── how_it_works.html
├── static/ # Optional: CSS, JS, Images
├── uploads/ # Temporary uploaded files (excluded in .gitignore)
├── requirements.txt
└── README.md

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
2. (Optional) Create a Virtual Environment
bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
3. Install Dependencies
bash
pip install -r requirements.txt
4. Run the Flask App
bash
python app.py
5. Open in Browser
http://127.0.0.1:5000
✅ Models Required
Ensure the following trained models are placed inside the models/ directory:

eye_movement_trained.h5

handwriting_model.h5 OR handwriting_dyslexia_model.pkl

audio_model.pkl

audio_scaler.pkl

🔧 Technologies Used
Python 3

Flask (for backend)

TensorFlow / Keras (for CNN models)

scikit-learn (for audio classification)

OpenCV, librosa, pandas, matplotlib

📌 Disclaimer
This tool is an early-stage research project intended for educational and experimental purposes only. It is not a medical diagnostic and should not be used as a substitute for professional evaluation.

👩‍💻 Developed by
Arige Nikhila and Bhogapurapu Susmitha

🏫 Velegapudi Ramakrishna Siddhartha Engineering College


📧 nikhilaarige54@gmail.com
```
