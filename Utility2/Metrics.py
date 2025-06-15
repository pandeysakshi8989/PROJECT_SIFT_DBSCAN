import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate accuracy, precision, recall, and F1-score from true and predicted labels.

    Parameters:
    - true_labels: Ground truth binary labels (1 for forged, 0 for authentic)
    - predicted_labels: Predicted binary labels

    Returns:
    - A dictionary with accuracy, precision, recall, and F1-score
    """
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Confusion matrix: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    return {
        "True Positives": int(tp),
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4)
    }

# Example usage
if __name__ == "__main__":
    # Sample true and predicted labels
    y_true = [1, 0, 1, 1, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 1]

    results = calculate_metrics(y_true, y_pred)
    for k, v in results.items():
        print(f"{k}: {v}")
