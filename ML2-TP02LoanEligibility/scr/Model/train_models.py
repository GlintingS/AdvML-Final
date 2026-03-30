from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Function to train the Logistic Regression Model
def train_LRmodel(X, y):
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

    # Save the trained model
    with open(MODELS_DIR / "LRmodel.pkl", "wb") as f:
        pickle.dump(model, f)

    return LRmodel, X_test_scaled, y_test


# Function to train the Random Forest Classifier Model
def train_RFmodel(X, y):
    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create an instance of the model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    RFmodel = rf.fit(x_train, y_train)

    # Save the trained model
    with open(MODELS_DIR / "RFmodel.pkl", "wb") as f:
        pickle.dump(RFmodel, f)

    return RFmodel, x_test, y_test
