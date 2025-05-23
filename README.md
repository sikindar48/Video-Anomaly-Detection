# Anomaly Detection System for Surveillance Videos
A GUI-based anomaly detection system for surveillance videos. 
Uses machine learning (SVM) and deep learning (BiLSTM) to classify video frames.  
Detects anomalies like fights or accidents in real-time.  
Built with Python, OpenCV, Tkinter, and PyTorch.

---

## 🔧 Features

- **Tkinter GUI** for dataset upload, model training, prediction & results
- **SVM & BiLSTM** models to compare performance
- **Real-time video analysis** with bounding boxes & decision labels
- **Accuracy, Precision, Recall, F1-score** evaluation with confusion matrix
- **Graphical comparison** between SVM and BiLSTM metrics

---

## 📁 Dataset Preparation

- Upload image datasets categorized into "InappropriateContent" and others.
- All images are resized to 32×32 and normalized.
- Dataset is automatically split into training and testing sets.

---

## 🚀 Prediction Flow

1. Load a video file via the GUI.
2. Each frame is preprocessed and analyzed.
3. SVM model classifies frames; majority voting gives final prediction.
4. Bounding box and classification label shown on video.
5. Results displayed in GUI and final screen.

---

## 📊 Evaluation Metrics

| Metric      | Description                                      |
|-------------|--------------------------------------------------|
| Accuracy    | Correct predictions / Total samples              |
| Precision   | True Positives / (True Positives + False Positives) |
| Recall      | True Positives / (True Positives + False Negatives) |
| F1-Score    | Harmonic mean of Precision and Recall            |

- Includes comparison graph for SVM and BiLSTM models.

---

## 🛠 Tech Stack

- **Python**  - Programming Language 
- **OpenCV** – Video processing  
- **Tkinter** – GUI interface  
- **PyTorch** – Deep learning (BiLSTM)  
- **Scikit-learn** – SVM & evaluation metrics  
- **Matplotlib, Seaborn** – Visualization  

---

## 🖥 How to Run

1. Clone this repository  
   ```bash
   git clone https://github.com/yourusername/AnomalyDetectionGUI.git
   cd AnomalyDetectionGUI

2.	Install requirements

   pip install -r requirements.txt

3.	Run the application

   python main.py

⸻

📌 Note
	•	Only video files are used for prediction.
	•	Image datasets are used only for training models.
	•	For custom classification, filename prefixes like fi, v13.mp4, etc., are recognized.

⸻

📷 Screenshots

<img width="511" alt="image" src="https://github.com/user-attachments/assets/237b27a5-928b-4093-9791-af9e56db0e62" />
<img width="511" alt="image" src="https://github.com/user-attachments/assets/ccc9cf61-1ce0-4f56-99e0-393ec3ac75d3" />
<img width="502" alt="image" src="https://github.com/user-attachments/assets/fbb7d159-6cb0-47ec-909f-c265741783d5" />
<img width="511" alt="image" src="https://github.com/user-attachments/assets/a6294c7f-c28a-457b-89d8-585cc073d3bd" />


⸻

📩 Contact

Nayab Sikindar
📧 nayabsikindar48@gmail.com
🔗 LinkedIn
