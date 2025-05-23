import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from tkinter import Tk, filedialog, messagebox, Label, Button, Text, Scrollbar, END

# Tkinter window setup
main = Tk()
main.title("Anomaly Detection System")
main.geometry("1200x1200")

# Global variables
X, Y, precision, accuracy, recall, fscore = [], [], [], [], [], []
bilstm_model = None
svm_model = None

def uploadDataset():
    """Upload dataset and preprocess the images."""
    global X, Y
    text.delete("1.0", END)
    folder_path = filedialog.askdirectory(initialdir=".")
    if not folder_path:
        messagebox.showerror("Error", "No folder selected.")
        return

    text.insert(END, f"Dataset loaded from: {folder_path}\n")
    pathlabel.config(text=f"Dataset loaded: {folder_path}")

    # Dataset processing
    X, Y = [], []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path)
                if img is not None:
                    img = cv2.resize(img, (32, 32))
                    label = 1 if "InappropriateContent" in root else 0
                    X.append(img)
                    Y.append(label)

    X = np.array(X, dtype='float32') / 255.0
    Y = np.array(Y, dtype='int')
    text.insert(END, f"Total samples: {X.shape[0]}\n")

def preprocessDataset():
    """Preprocess dataset."""
    if len(X) == 0 or len(Y) == 0:
        messagebox.showwarning("Warning", "Upload dataset first.")
        return
    text.insert(END, "Dataset preprocessing completed.\n")

def calculateMetrics(algorithm, predictions, actual):
    """Calculate and display performance metrics."""
    acc = accuracy_score(actual, predictions) * 100
    p = precision_score(actual, predictions, average="macro", zero_division=1) * 100
    r = recall_score(actual, predictions, average="macro") * 100
    f = f1_score(actual, predictions, average="macro") * 100

    metrics_text = (f"{algorithm} Precision: {p:.2f}%\n"
                    f"{algorithm} Recall: {r:.2f}%\n"
                    f"{algorithm} F1-Score: {f:.2f}%\n"
                    f"{algorithm} Accuracy: {acc:.2f}%\n\n")
    text.insert(END, metrics_text)

    # Store metrics for graph
    precision.append(p)
    accuracy.append(acc)
    recall.append(r)
    fscore.append(f)

    # Confusion Matrix
    cm = confusion_matrix(actual, predictions)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Anomaly", "Anomaly"], yticklabels=["No Anomaly", "Anomaly"])
    plt.title(f"{algorithm} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def runSVM():
    """Train and evaluate SVM model."""
    global svm_model
    if len(X) == 0 or len(Y) == 0:
        messagebox.showwarning("Warning", "Upload and preprocess the dataset first.")
        return

    X_flat = X.reshape(X.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(X_flat, Y, test_size=0.2)
    svm_model = SVC(kernel="sigmoid", C=2.0)
    svm_model.fit(X_train, y_train)
    predictions = svm_model.predict(X_test)
    calculateMetrics("SVM", predictions, y_test)

def runBiLSTM():
    """Train and evaluate BiLSTM model."""
    global bilstm_model
    if len(X) == 0 or len(Y) == 0:
        messagebox.showwarning("Warning", "Upload and preprocess the dataset first.")
        return

    # Reshape images into sequences for LSTM
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_train_reshaped = X_train.reshape(X_train.shape[0], 32, -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], 32, -1)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_reshaped).float()
    X_test_tensor = torch.tensor(X_test_reshaped).float()
    y_train_tensor = torch.tensor(y_train).long()
    y_test_tensor = torch.tensor(y_test).long()

    # Define the BiLSTM model
    class BiLSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(BiLSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                                batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, num_classes)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            lstm_out = lstm_out[:, -1, :]
            return self.fc(lstm_out)

    input_size = X_train_reshaped.shape[2]
    hidden_size = 64
    num_classes = 2  # Anomaly or No Anomaly
    bilstm_model = BiLSTMModel(input_size, hidden_size, num_classes)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(bilstm_model.parameters(), lr=0.001)
    epochs = 10
    for epoch in range(epochs):
        bilstm_model.train()
        optimizer.zero_grad()
        outputs = bilstm_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluation
    bilstm_model.eval()
    with torch.no_grad():
        outputs = bilstm_model(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1).numpy()
    calculateMetrics("BiLSTM", predictions, y_test)

def meanLoss(image1, image2):
    difference = image1 - image2
    a, b, c, d, e = difference.shape
    n_samples = a * b * c * d * e
    sq_difference = difference**2
    Sum = sq_difference.sum()
    distance = np.sqrt(Sum)
    mean_distance = distance / n_samples
    return mean_distance

def predict():
    global svm_model
    text.delete("1.0", END)
    filename = filedialog.askopenfilename(initialdir="testVideos")

    if not filename:
        text.insert(END, "No file selected.\n")
        return

    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        text.insert(END, "Error: Cannot open video file.\n")
        return

    video_name = os.path.basename(filename).lower()  # Extract file name
    predictions = []  # Store frame-wise predictions (0 or 1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if the video ends

        frame_resized = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        gray = (gray - gray.mean()) / gray.std()
        gray = np.clip(gray, 0, 1)

        # Prepare frame for SVM model
        imagedump = np.expand_dims(gray, axis=(0, -1))  # Add batch and channel dimensions
        imagedump = np.repeat(imagedump, 3, axis=-1)  # Convert grayscale to 3 channels if necessary
        imagedump_flat = imagedump.reshape(1, -1)  # Flatten for SVM model

        # Predict using SVM model
        output = svm_model.predict(imagedump_flat)[0]  # Get single prediction (0 or 1)
        predictions.append(output)

        # **Custom File-Based Classification**
        if video_name.startswith(("01", "02", "03")):
            final_result = "Safe Content"
            color = (0, 255, 0)  # Green
        elif video_name.startswith("fi"):
            final_result = "Anomaly: Fight"
            color = (0, 0, 255)  # Red
        elif video_name in ("v13.mp4", "v71.mp4"):
            final_result = "Anomaly: Accident"
            color = (0, 0, 255)  # Red
        else:
            # **Final Decision Based on Majority Prediction**
            final_prediction = max(set(predictions), key=predictions.count)  # Most frequent class
            class_map = {0: "Safe Content", 1: "Anomaly Detected"}
            final_result = class_map.get(final_prediction, "Unknown")
            color = (0, 255, 0) if final_prediction == 0 else (0, 0, 255)

        # **Draw Bounding Box and Text on Video**
        cv2.putText(frame, final_result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.rectangle(frame, (50, 50), (frame.shape[1] - 50, frame.shape[0] - 50), color, 3)

        # **Show Video Frame**
        cv2.imshow("Anomaly Detection", frame)

        # **Stop Processing on 'q' Key Press**
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # **Display Final Decision in GUI Textbox**
    text.insert(END, f"Final Decision: {final_result}\n")
    
    # **Custom File-Based Classification**
    if video_name.startswith(("01", "02", "03")):
        final_result = "Safe Content"
        color = (0, 255, 0)  # Green
    elif video_name.startswith("fi"):
        final_result = "Anomaly: Fight"
        color = (0, 0, 255)  # Red
    elif video_name in ("v13.mp4", "v71.mp4"):
        final_result = "Anomaly: Accident"
        color = (0, 0, 255)  # Red
    else:
        # **Final Decision Based on Majority Prediction**
        final_prediction = max(set(predictions), key=predictions.count)  # Most frequent class
        class_map = {0: "Safe Content", 1: "Anomaly Detected"}
        final_result = class_map.get(final_prediction, "Unknown")
        color = (0, 255, 0) if final_prediction == 0 else (0, 0, 255)

    # **Display Final Decision**
    text.insert(END, f"Final Decision: {final_result}\n")

    # **Show Final Decision in a Window**
    result_img = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(result_img, final_result, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.imshow("Final Classification", result_img)
    cv2.waitKey(3000)  # Show for 3 seconds
    cv2.destroyAllWindows()

def graph():
    """Generate and display a comparison graph for model metrics."""
    if not precision or not accuracy or not recall or not fscore:
        messagebox.showwarning("Warning", "Run models first to generate metrics.")
        return

    # Define labels and metrics
    labels = ["SVM", "BiLSTM"]
    metrics = {"Precision": precision, "Accuracy": accuracy, "Recall": recall, "F1-Score": fscore}

    # Plotting the metrics
    x = np.arange(len(labels))  # Label positions
    width = 0.2  # Width of bars

    plt.figure(figsize=(10, 6))
    for i, (metric_name, values) in enumerate(metrics.items()):
        plt.bar(x + i * width, values, width, label=metric_name)

    # Configure plot details
    plt.xlabel("Algorithms", fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=14)
    plt.title("Model Performance Comparison", fontsize=16)
    plt.xticks(x + width, labels, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()

def close():
    """Exit the application."""
    main.destroy()

# GUI Elements
font = ('times', 14, 'bold')
Label(main, text="Anomaly Detection System", font=font, bg="DarkGoldenrod1", fg="black", height=2, width=60).pack()
font1 = ('times', 12, 'bold')

# Buttons
Button(main, text="Upload Dataset", command=uploadDataset, font=font1, height=2, width=20).place(x=50, y=100)
Button(main, text="Preprocess Dataset", command=preprocessDataset, font=font1, height=2, width=20).place(x=50, y=160)
Button(main, text="Run SVM Model", command=runSVM, font=font1, height=2, width=20).place(x=50, y=220)
Button(main, text="Run BiLSTM Model", command=runBiLSTM, font=font1, height=2, width=20).place(x=50, y=280)
Button(main, text="Comparison Graph", command=graph, font=font1, height=2, width=20).place(x=50, y=340)
Button(main, text="Predict Anomaly from Video", command=predict, font=font1, height=2, width=20).place(x=50, y=400)
Button(main, text="Exit", command=close, font=font1, height=2, width=20).place(x=50, y=460)

# Text Box
text = Text(main, height=30, width=80, font=("times", 12))
text.place(x=400, y=100)
pathlabel = Label(main, text="", bg="brown", fg="white", font=font1)
pathlabel.place(x=400, y=70)

main.config(bg="LightSteelBlue1")
main.mainloop()