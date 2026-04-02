import logging

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def load_and_preprocess_data(data_path):
    project_root = Path(__file__).resolve().parents[2]

    # Import the data from raw data file
    logger.info("Loading raw data from %s", data_path)
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("Data file not found: %s", data_path)
        raise
    except pd.errors.ParserError as exc:
        logger.error("Failed to parse CSV file: %s", exc)
        raise

    logger.info("Loaded %d rows and %d columns", len(df), len(df.columns))

    # Handle missing values - fill with mode for each column
    for col in df.columns:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
                logger.debug(
                    "Filled missing values in '%s' with mode: %s", col, mode_val[0]
                )

    # Store the target variable in y (Loan_Approved: Y/N)
    if "Loan_Approved" not in df.columns:
        logger.error("Target column 'Loan_Approved' not found in dataset")
        raise KeyError("Target column 'Loan_Approved' not found in dataset")
    y = df["Loan_Approved"].map({"Y": 1, "N": 0})

    # Separate input features in x - drop non-predictive columns
    x = df.drop(["Loan_ID", "Loan_Approved"], axis=1)

    # Encode categorical variables
    categorical_columns = x.select_dtypes(include=["object"]).columns
    label_encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col].astype(str))
        label_encoders[col] = le

    logger.info("Encoded %d categorical columns", len(categorical_columns))

    # Save the cleaned data
    output_path = project_root / "data" / "processed" / "cleaned_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df = df.copy()
    cleaned_df["Loan_Approved"] = y
    try:
        cleaned_df.to_csv(output_path, index=False)
        logger.info("Saved cleaned data to %s", output_path)
    except OSError as exc:
        logger.error("Failed to save cleaned data: %s", exc)
        raise

    return df, x, y
