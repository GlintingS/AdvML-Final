import logging
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Function to train the Logistic Regression Model
def train_LRmodel(X, y):
    logger.info("Training Logistic Regression model")
    try:
        # Splitting the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale the data using StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(x_train)
        X_test_scaled = scaler.transform(x_test)

        # Train your model
        model = LogisticRegression(random_state=42, max_iter=1000)
        LRmodel = model.fit(X_train_scaled, y_train)
        logger.info("Logistic Regression model trained successfully")

        # Save the trained model
        with open(MODELS_DIR / "LRmodel.pkl", "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved LR model to %s", MODELS_DIR / "LRmodel.pkl")

        return LRmodel, X_test_scaled, y_test
    except Exception as exc:
        logger.error("Failed to train Logistic Regression model: %s", exc)
        raise


# Function to train the Random Forest Classifier Model
def train_RFmodel(X, y):
    logger.info("Training Random Forest Classifier model")
    try:
        # Splitting the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create an instance of the model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train the model
        RFmodel = rf.fit(x_train, y_train)
        logger.info("Random Forest model trained successfully")

        # Save the trained model
        with open(MODELS_DIR / "RFmodel.pkl", "wb") as f:
            pickle.dump(RFmodel, f)
        logger.info("Saved RF model to %s", MODELS_DIR / "RFmodel.pkl")

        return RFmodel, x_test, y_test
    except Exception as exc:
        logger.error("Failed to train Random Forest model: %s", exc)
        raise
