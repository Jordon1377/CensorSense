from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import tensorflow as tf

# Function to calculate metrics (TP, TN, FP, FN)
def calculate_metrics(y_true, y_pred):
    # Assume y_true and y_pred are numpy arrays
    # Confusion matrix: [TP, FP], [FN, TN]
    cm = confusion_matrix(y_true, y_pred)
    
    # Extracting TP, FP, FN, TN
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]
    
    return TP, FP, FN, TN

# Function to load a saved model and evaluate on a test set
def evaluate_model(model_path, test_images, test_labels):
    model = tf.keras.models.load_model(model_path)

    # Predict on test data
    predictions = model.predict(test_images)

    # For simplicity, assume binary classification: threshold prediction at 0.5
    predictions = (predictions > 0.5).astype(int)

    # Calculate metrics
    TP, FP, FN, TN = calculate_metrics(test_labels, predictions)

    # Compute additional metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # Print confusion matrix and classification report
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print("\nClassification Report:\n", classification_report(test_labels, predictions))

# Example usage (assume you have test data ready)
# test_images and test_labels are the test set (e.g., images and corresponding true labels)

# Evaluate after training the model
model_path = "Model/pnet_epoch_8_batch_8000.h5"  # Change to the path of the saved model
evaluate_model(model_path, test_images, test_labels)