import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tkinter as tk
from tkinter import scrolledtext, Entry, Label, Button, messagebox, ttk
import matplotlib.pyplot as plt

# Function to extract MFCC features from audio files
def extract_features(audio_paths, sample_rate=22050, n_mfcc=13):
    features = []
    for audio_path in audio_paths:
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=sample_rate)

            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

            # Flatten the MFCC matrix and take the mean along each column to get a feature vector
            feature_vector = np.mean(mfccs.T, axis=0)
            features.append(feature_vector)
        except Exception as e:
            print("Skipping audio file:", audio_path)
            print("Error:", e)
    return features

# Prepare data and labels
def prepare_data(data_path):
    data = []
    labels = []
    class_folders = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))]

    for i, folder in enumerate(class_folders):
        class_dir = os.path.join(data_path, folder)
        audio_files = [os.path.join(class_dir, file) for file in os.listdir(class_dir) if file.endswith(('.wav', '.mp3', '.ogg'))]
        features = extract_features(audio_files)
        data.extend(features)
        labels.extend([i] * len(features))

    data = np.array(data)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

# Define classifiers and their parameters
classifiers = {
    "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), solver='lbfgs', random_state=42),
    "KNeighbors Classifier": KNeighborsClassifier(n_neighbors=3),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM Classifier": SVC(kernel='rbf', gamma='scale', random_state=42),
    "Naive Bayes Classifier": GaussianNB(),
    "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "XGBoost Classifier": XGBClassifier(random_state=42),
    "Gaussian Mixture Model": GaussianMixture(),
    "Decision Tree Classifier": DecisionTreeClassifier(),  # Added Decision Tree Classifier
    "Gaussian Process Classifier": GaussianProcessClassifier()  # Added Gaussian Process Classifier
}

# Train and evaluate classifiers
def train_and_evaluate_classifiers(X_train, X_test, y_train, y_test, classifiers):
    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        
        # Train dataset metrics
        y_train_pred = clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_report = classification_report(y_train, y_train_pred, output_dict=True)
        train_confusion = confusion_matrix(y_train, y_train_pred)
        
        # Test dataset metrics
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        confusion = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "train_accuracy": train_accuracy,
            "train_classification_report": train_report,
            "train_confusion_matrix": train_confusion,
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": confusion,
            "predicted_labels": y_pred
        }
    return results

# Function to display results in a pop-up window
def display_results(results, class_labels):
    def show_result(event):
        selection = algorithm_var.get()
        if selection == "All Algorithms":
            display_all_algorithms()
        else:
            result_text = f"{selection} Results:\n"
            
            # Train metrics
            result_text += "Train Metrics:\n"
            result_text += f"Accuracy: {results[selection]['train_accuracy']:.4f}\n"
            result_text += "Classification Report:\n"
            result_text += classification_report(y_train, classifiers[selection].predict(X_train)) + "\n"
            result_text += "Confusion Matrix:\n"
            result_text += np.array2string(results[selection]['train_confusion_matrix']) + "\n\n"
            
            # Test metrics
            result_text += "Test Metrics:\n"
            result_text += f"Accuracy: {results[selection]['accuracy']:.4f}\n"
            result_text += "Classification Report:\n"
            result_text += classification_report(y_test, results[selection]['predicted_labels']) + "\n"
            result_text += "Confusion Matrix:\n"
            result_text += np.array2string(results[selection]['confusion_matrix']) + "\n\n"
            
            text_area.delete(1.0, tk.END)
            text_area.insert(tk.INSERT, result_text)

    def detect_voice():
        file_path = entry.get()
        if not os.path.isfile(file_path):
            messagebox.showerror("Error", "File not found!")
            return

        feature = extract_features([file_path])
        if not feature:
            messagebox.showerror("Error", "Could not extract features from the file!")
            return

        feature = scaler.transform(feature)
        selected_classifier = algorithm_var.get()

    # Check if the selected classifier is 'All Algorithms'
        if selected_classifier == 'All Algorithms':
            display_all_algorithms()  # Call the function to display all algorithm results
            return

    # Check if the selected classifier exists in the classifiers dictionary
        if selected_classifier not in classifiers:
            messagebox.showerror("Error", f"Selected classifier '{selected_classifier}' is not valid!")
            return

        classifier = classifiers[selected_classifier]
        prediction = classifier.predict(feature)

        class_name = class_labels[prediction[0]]
        if class_name == "Intrusive":
            messagebox.showwarning("Warning", f"The file is classified as: {class_name}")
        elif class_name == "Non-intrusive":
            messagebox.showinfo("Info", f"The file is classified as: {class_name}. It seems to be an underwater recording. Enjoy!")
        else:
            messagebox.showinfo("Info", f"The file is classified as: {class_name}")

    def display_all_algorithms():
        root = tk.Tk()
        root.title("All Algorithms Results")
    
    # Calculate height of the table based on the number of algorithms
        num_algorithms = len(results)
        table_height = min(num_algorithms * 25, 500)  # Set a maximum height of 500 pixels

    # Set window size
        root.geometry(f"1000x{table_height}")

        tree = ttk.Treeview(root, columns=("Algorithm", "Train Accuracy", "Train Precision", "Train Recall", "Train F1 Score", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score"), show="headings")
        tree.heading("Algorithm", text="Algorithm")
        tree.heading("Train Accuracy", text="Train Accuracy")
        tree.heading("Train Precision", text="Train Precision")
        tree.heading("Train Recall", text="Train Recall")
        tree.heading("Train F1 Score", text="Train F1 Score")
        tree.heading("Test Accuracy", text="Test Accuracy")
        tree.heading("Test Precision", text="Test Precision")
        tree.heading("Test Recall", text="Test Recall")
        tree.heading("Test F1 Score", text="Test F1 Score")

    # Set the width of each column
        tree.column("Algorithm", width=200)
        tree.column("Train Accuracy", width=100)
        tree.column("Train Precision", width=100)
        tree.column("Train Recall", width=100)
        tree.column("Train F1 Score", width=100)
        tree.column("Test Accuracy", width=100)
        tree.column("Test Precision", width=100)
        tree.column("Test Recall", width=100)
        tree.column("Test F1 Score", width=100)

    # Lists to store metrics for averaging
        train_accuracies = []
        train_precisions = []
        train_recalls = []
        train_f1_scores = []
        test_accuracies = []
        test_precisions = []
        test_recalls = []
        test_f1_scores = []

    for name, result in results.items():
        train_accuracy = result['train_accuracy']
        train_report = result['train_classification_report']
        train_precision = train_report['weighted avg']['precision']
        train_recall = train_report['weighted avg']['recall']
        train_f1_score = train_report['weighted avg']['f1-score']
        
        test_accuracy = result['accuracy']
        test_report = result['classification_report']
        test_precision = test_report['weighted avg']['precision']
        test_recall = test_report['weighted avg']['recall']
        test_f1_score = test_report['weighted avg']['f1-score']
        
        tree.insert("", "end", values=(name, train_accuracy, train_precision, train_recall, train_f1_score, test_accuracy, test_precision, test_recall, test_f1_score))

        # Append metrics for averaging
        train_accuracies.append(train_accuracy)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1_scores.append(train_f1_score)
        test_accuracies.append(test_accuracy)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1_scores.append(test_f1_score)

    # Calculate average metrics
    avg_train_accuracy = np.mean(train_accuracies)
    avg_train_precision = np.mean(train_precisions)
    avg_train_recall = np.mean(train_recalls)
    avg_train_f1_score = np.mean(train_f1_scores)
    avg_test_accuracy = np.mean(test_accuracies)
    avg_test_precision = np.mean(test_precisions)
    avg_test_recall = np.mean(test_recalls)
    avg_test_f1_score = np.mean(test_f1_scores)

    # Insert row for average metrics
    tree.insert("", "end", values=("Average", avg_train_accuracy, avg_train_precision, avg_train_recall, avg_train_f1_score, avg_test_accuracy, avg_test_precision, avg_test_recall, avg_test_f1_score))

    # Set the font size of the table
    style = ttk.Style()
    style.configure("Treeview", font=("Arial", 12))

    tree.pack(fill=tk.BOTH, expand=True)

    root.mainloop()


    root = tk.Tk()
    root.title("Classifier Results")

    frame = tk.Frame(root)
    frame.pack(pady=20, padx=20)

    algorithm_var = tk.StringVar()
    algorithm_var.set(list(results.keys())[0])
    algorithm_options = list(results.keys()) + ["All Algorithms"]
    algorithm_dropdown = tk.OptionMenu(frame, algorithm_var, *algorithm_options, command=show_result)
    algorithm_dropdown.config(width=25)
    algorithm_dropdown.pack(side=tk.LEFT, padx=10, pady=10)

    text_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=80, height=30, font=("Times New Roman", 12))
    text_area.pack(side=tk.RIGHT, padx=10, pady=10)

    entry_label = Label(root, text="Enter audio file path:")
    entry_label.pack(pady=5)
    entry = Entry(root, width=50)
    entry.pack(pady=5)

    detect_button = Button(root, text="Detect Voice", command=detect_voice)
    detect_button.pack(pady=5)

    root.mainloop()

def plot_metrics(results):
    algorithms = list(results.keys())
    accuracy_scores = [results[alg]['accuracy'] for alg in algorithms]
    precision_scores = [results[alg]['classification_report']['weighted avg']['precision'] for alg in algorithms]
    recall_scores = [results[alg]['classification_report']['weighted avg']['recall'] for alg in algorithms]
    f1_scores = [results[alg]['classification_report']['weighted avg']['f1-score'] for alg in algorithms]

    # Plotting Accuracy
    plt.figure(figsize=(8, 6))
    plt.bar(algorithms, accuracy_scores, color='skyblue')
    plt.title('Accuracy Scores of Classifiers')
    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Plotting Precision
    plt.figure(figsize=(8, 6))
    plt.bar(algorithms, precision_scores, color='salmon')
    plt.title('Precision Scores of Classifiers')
    plt.xlabel('Classifiers')
    plt.ylabel('Precision')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Plotting Recall
    plt.figure(figsize=(8, 6))
    plt.bar(algorithms, recall_scores, color='lightgreen')
    plt.title('Recall Scores of Classifiers')
    plt.xlabel('Classifiers')
    plt.ylabel('Recall')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Plotting F1 Score
    plt.figure(figsize=(8, 6))
    plt.bar(algorithms, f1_scores, color='orange')
    plt.title('F1 Scores of Classifiers')
    plt.xlabel('Classifiers')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_path = r"C:\Users\vaish\Downloads\resource_map_doi_10_5065_D66Q1VB7\data\pl1"
    X_train, X_test, y_train, y_test, scaler = prepare_data(data_path)
    results = train_and_evaluate_classifiers(X_train, X_test, y_train, y_test, classifiers)

    class_labels = {0: "Non-intrusive", 1: "Intrusive"}  # Assuming binary classification

    print("Class Counts:")
    class_counts = {}
    for folder in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, folder)):
            class_dir = os.path.join(data_path, folder)
            num_files = len([file for file in os.listdir(class_dir) if file.endswith(('.wav', '.mp3', '.ogg'))])
            class_counts[folder] = num_files
            print(f"{folder}: {num_files} files")

    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    if max_count - min_count > 0.5 * (min_count + max_count):
        print("The dataset is imbalanced.")
    else:
        print("The dataset is balanced.")

    root = tk.Tk()
    root.title("Classifier Results")

    display_results(results, class_labels, root)
    plot_metrics(results)
