import logging
import sys
from pathlib import Path

from scr.data import make_dataset
from scr.Model import predict_models, train_models
from scr.visuals import visualize

# Configure logging for the entire application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Loan Eligibility pipeline")

    try:
        # Load and preprocess the raw data
        root = Path(__file__).resolve().parent
        data_path = root / "data" / "raw" / "credit.csv"

        # Preprocess the raw data
        df, X, y = make_dataset.load_and_preprocess_data(str(data_path))

        # Train the Logistic Regression model
        LRmodel, X_LR_test, y_LR_test = train_models.train_LRmodel(X, y)

        # Train the Random Forest Classifier model
        RFmodel, X_RF_test, y_RF_test = train_models.train_RFmodel(X, y)

        # Evaluate the models - get accuracy, precision, recall, f1
        LR_acc, LR_prec, LR_rec, LR_f1 = predict_models.evaluate_model(
            LRmodel, X_LR_test, y_LR_test
        )
        RF_acc, RF_prec, RF_rec, RF_f1 = predict_models.evaluate_model(
            RFmodel, X_RF_test, y_RF_test
        )

        # Print evaluation results
        print("\n=== Loan Eligibility Classification Results ===\n")
        print("Logistic Regression:")
        print(f"  Accuracy:  {LR_acc:.4f}")
        print(f"  Precision: {LR_prec:.4f}")
        print(f"  Recall:    {LR_rec:.4f}")
        print(f"  F1-Score:  {LR_f1:.4f}")

        print("\nRandom Forest Classifier:")
        print(f"  Accuracy:  {RF_acc:.4f}")
        print(f"  Precision: {RF_prec:.4f}")
        print(f"  Recall:    {RF_rec:.4f}")
        print(f"  F1-Score:  {RF_f1:.4f}")

        # Plot the accuracy values for both models
        models = ["Logistic Regression", "Random Forest"]
        accuracy_values = [LR_acc, RF_acc]

        visualize.plot_mae(models, accuracy_values)

        logger.info("Pipeline completed successfully")
    except FileNotFoundError as exc:
        logger.error("Required file not found: %s", exc)
        sys.exit(1)
    except KeyError as exc:
        logger.error("Data issue — missing expected column: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Pipeline failed with unexpected error: %s", exc, exc_info=True)
        sys.exit(1)
