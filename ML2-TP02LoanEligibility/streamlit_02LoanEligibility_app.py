import logging
from pathlib import Path
import pickle
from typing import Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    handlers=[logging.FileHandler("streamlit_app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "RFmodel.pkl"
LR_MODEL_PATH = PROJECT_ROOT / "models" / "LRmodel.pkl"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit.csv"

st.set_page_config(page_title="Loan Eligibility Predictor", page_icon="💰")

# Set the page title and description
st.title("💰 Loan Eligibility Predictor")
st.write(
    """
    This app predicts loan eligibility using machine learning models.
    Enter applicant details below to check loan approval chances.
    """
)


@st.cache_resource
def load_model_from_bytes(model_bytes: bytes):
    return pickle.loads(model_bytes)


def get_model():
    if MODEL_PATH.exists():
        logger.info("Loading Random Forest model from %s", MODEL_PATH)
        return (
            load_model_from_bytes(MODEL_PATH.read_bytes()),
            f"Loaded Random Forest model from {MODEL_PATH}",
        )

    logger.warning("No local model found at %s", MODEL_PATH)
    st.warning(
        "No local model found at models/RFmodel.pkl. Upload a trained model file to continue."
    )
    uploaded_model = st.file_uploader("Upload Random Forest model (.pkl)", type=["pkl"])
    if uploaded_model is None:
        return None, "Model not loaded"

    try:
        return load_model_from_bytes(uploaded_model.getvalue()), "Loaded uploaded model"
    except Exception as exc:
        logger.error("Could not read uploaded model file: %s", exc)
        st.error(f"Could not read uploaded model file: {exc}")
        return None, "Failed to load uploaded model"


@st.cache_data
def load_reference_data() -> tuple[Optional[pd.DataFrame], str]:
    if PROCESSED_DATA_PATH.exists():
        return (
            pd.read_csv(PROCESSED_DATA_PATH),
            f"Loaded data from {PROCESSED_DATA_PATH}",
        )
    if RAW_DATA_PATH.exists():
        return pd.read_csv(RAW_DATA_PATH), f"Loaded data from {RAW_DATA_PATH}"
    return None, "No local dataset found for evaluation"


def preprocess_input(
    input_df: pd.DataFrame, reference_df: pd.DataFrame
) -> pd.DataFrame:
    """Encode categorical variables to match training data"""
    processed = input_df.copy()

    categorical_cols = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]

    for col in categorical_cols:
        if col in processed.columns and col in reference_df.columns:
            le = LabelEncoder()
            # Fit on reference data to ensure consistent encoding
            unique_vals = reference_df[col].dropna().unique()
            le.fit(unique_vals)
            processed[col] = le.transform([processed[col].values[0]])[0]

    return processed


def build_evaluation_matrix(
    df: pd.DataFrame, model, lr_model
) -> tuple[Optional[pd.DataFrame], str]:
    """Build evaluation matrix with classification metrics"""
    needed_columns = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area",
        "Loan_Approved",
    ]

    missing_columns = [c for c in needed_columns if c not in df.columns]
    if missing_columns:
        return None, f"Missing required columns for evaluation: {missing_columns}"

    eval_df = df[needed_columns].dropna()
    if len(eval_df) < 50:
        return None, "Not enough valid rows to compute a stable evaluation matrix"

    # Prepare features
    X = eval_df.drop("Loan_Approved", axis=1)

    # Support both raw labels (Y/N) and already encoded labels (1/0)
    target_raw = eval_df["Loan_Approved"]
    if target_raw.dtype == object:
        y = target_raw.astype(str).str.strip().str.upper().map({"Y": 1, "N": 0})
    else:
        y = pd.to_numeric(target_raw, errors="coerce")

    # Remove rows where target could not be parsed
    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(int)

    if len(y) < 50:
        return None, "Not enough valid target rows to compute evaluation metrics"

    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    metric_rows = []

    if model is not None and hasattr(model, "predict"):
        y_pred_rf = model.predict(x_test)
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        metric_rows.append(
            {
                "Model": "Random Forest",
                "Accuracy": rf_accuracy,
                "Precision": precision_score(y_test, y_pred_rf, zero_division=0),
                "Recall": recall_score(y_test, y_pred_rf, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred_rf, zero_division=0),
            }
        )

    if lr_model is not None and hasattr(lr_model, "predict"):
        y_pred_lr = lr_model.predict(x_test)
        lr_accuracy = accuracy_score(y_test, y_pred_lr)
        metric_rows.append(
            {
                "Model": "Logistic Regression",
                "Accuracy": lr_accuracy,
                "Precision": precision_score(y_test, y_pred_lr, zero_division=0),
                "Recall": recall_score(y_test, y_pred_lr, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred_lr, zero_division=0),
            }
        )

    matrix_df = pd.DataFrame(metric_rows)
    return matrix_df, f"Computed on {len(eval_df)} samples (80/20 split)"


rf_model, model_message = get_model()
st.caption(model_message)

# Load LR model if available
lr_model = None
if LR_MODEL_PATH.exists():
    logger.info("Loading Logistic Regression model from %s", LR_MODEL_PATH)
    lr_model = load_model_from_bytes(LR_MODEL_PATH.read_bytes())

reference_df, data_message = load_reference_data()
st.caption(data_message)


# Prepare the form to collect user inputs
with st.form("loan_application"):
    st.subheader("📋 Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    with col2:
        applicant_income = st.number_input(
            "Applicant Income (₹)", min_value=0, step=1000
        )
        coapplicant_income = st.number_input(
            "Co-applicant Income (₹)", min_value=0, step=1000
        )
        loan_amount = st.number_input("Loan Amount (₹)", min_value=0, step=10000)
        loan_term = st.selectbox(
            "Loan Amount Term (months)", [360, 480, 300, 240, 180, 120, 60]
        )
        credit_history = st.selectbox("Credit History", [1.0, 0.0])

    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    # Submit button
    submitted = st.form_submit_button("🔍 Check Loan Eligibility")


if submitted:
    if rf_model is None:
        st.error("Model is not loaded. Add models/RFmodel.pkl or upload one.")
        st.stop()

    # Prepare input for prediction
    input_data = pd.DataFrame(
        {
            "Gender": [gender],
            "Married": [married],
            "Dependents": [dependents],
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "LoanAmount": [loan_amount],
            "Loan_Amount_Term": [loan_term],
            "Credit_History": [credit_history],
            "Property_Area": [property_area],
        }
    )

    # Encode categorical variables
    categorical_cols = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]
    for col in categorical_cols:
        le = LabelEncoder()
        unique_vals = (
            reference_df[col].dropna().unique() if reference_df is not None else []
        )
        if len(unique_vals) > 0:
            le.fit(unique_vals)
            input_data[col] = le.transform(input_data[col].astype(str))

    try:
        prediction = rf_model.predict(input_data)[0]
        prediction_proba = rf_model.predict_proba(input_data)[0]
        logger.info("Prediction: %s (prob=%.4f)", prediction, prediction_proba[1])
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        st.error(f"Prediction failed: {exc}")
        st.stop()

    # Display prediction result
    st.subheader("✅ Prediction Result")

    if prediction == 1:
        st.success(
            f"🎉 **LOAN APPROVED** - Approval Probability: {prediction_proba[1]:.2%}"
        )
    else:
        st.error(
            f"❌ **LOAN NOT APPROVED** - Approval Probability: {prediction_proba[1]:.2%}"
        )

    # Show prediction probabilities as a gauge
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Approval Chance", f"{prediction_proba[1]:.2%}")
    with col2:
        st.metric("Rejection Chance", f"{prediction_proba[0]:.2%}")


# Model Performance Section
st.divider()
st.subheader("📊 Model Evaluation Metrics")

if reference_df is None:
    st.info("Evaluation metrics are unavailable because no dataset was found.")
else:
    try:
        metrics_df, metrics_note = build_evaluation_matrix(
            reference_df, rf_model, lr_model
        )
        if metrics_df is None:
            st.warning(metrics_note)
        else:
            display_df = metrics_df.copy()
            for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].map(lambda v: f"{v:.4f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.caption(metrics_note)
    except Exception as exc:
        st.error(f"Could not compute evaluation metrics: {exc}")


# Feature Importance Section
st.divider()
st.subheader("🎯 Feature Importance")

if rf_model is not None and hasattr(rf_model, "feature_importances_"):
    feature_names = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area",
    ]

    if len(rf_model.feature_importances_) == len(feature_names):
        importances = pd.Series(
            rf_model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        importances.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Random Forest Feature Importance")
        st.pyplot(fig)
else:
    st.info("Feature importance is not available for the loaded model.")


# Dataset Insights
st.divider()
st.subheader("📈 Dataset Insights")

if reference_df is None or "Loan_Approved" not in reference_df.columns:
    st.info("Dataset insights are unavailable because no local data file was found.")
else:
    plot_df = reference_df.copy()

    # Loan approval distribution
    col1, col2 = st.columns(2)

    with col1:
        approval_counts = plot_df["Loan_Approved"].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
        ax_pie.pie(
            approval_counts,
            labels=["Approved", "Not Approved"],
            autopct="%1.1f%%",
            colors=["#2ecc71", "#e74c3c"],
        )
        ax_pie.set_title("Loan Approval Distribution")
        st.pyplot(fig_pie)

    with col2:
        # Income distribution
        fig_income, ax_income = plt.subplots(figsize=(8, 4))
        plot_df["ApplicantIncome"].hist(
            bins=30, ax=ax_income, color="skyblue", edgecolor="black"
        )
        ax_income.set_title("Applicant Income Distribution")
        ax_income.set_xlabel("Income (₹)")
        ax_income.set_ylabel("Frequency")
        st.pyplot(fig_income)

st.write(
    """
    **How to use this app:**
    1. Fill in the applicant details in the form
    2. Click 'Check Loan Eligibility' to get a prediction
    3. View model performance metrics and feature importance
    4. Check dataset insights for loan approval patterns
    """
)
