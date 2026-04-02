import logging

# Import classification metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


# Function to predict and evaluate (classification)
def evaluate_model(model, X_test, y_test):
    """
    Evaluate classification model using multiple metrics
    Returns: accuracy, precision, recall, f1
    """
    logger.info("Evaluating model: %s", type(model).__name__)
    try:
        # Make predictions on test set
        y_pred = model.predict(X_test)

        # Calculate classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        logger.info("Evaluation complete — Accuracy: %.4f, F1: %.4f", accuracy, f1)
        return accuracy, precision, recall, f1
    except Exception as exc:
        logger.error("Model evaluation failed: %s", exc)
        raise


def get_confusion_matrix(model, X_test, y_test):
    """Get confusion matrix for the model"""
    try:
        y_pred = model.predict(X_test)
        return confusion_matrix(y_test, y_pred)
    except Exception as exc:
        logger.error("Failed to compute confusion matrix: %s", exc)
        raise


def get_classification_report(model, X_test, y_test):
    """Get detailed classification report"""
    try:
        y_pred = model.predict(X_test)
        return classification_report(
            y_test, y_pred, target_names=["Not Approved", "Approved"]
        )
    except Exception as exc:
        logger.error("Failed to generate classification report: %s", exc)
        raise
