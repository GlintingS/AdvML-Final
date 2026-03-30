import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(data_path):
    project_root = Path(__file__).resolve().parents[2]

    # Import the data from raw data file
    df = pd.read_csv(data_path)

    # Handle missing values - fill with mode for each column
    for col in df.columns:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])

    # Store the target variable in y (Loan_Approved: Y/N)
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

    # Save the cleaned data
    output_path = project_root / "data" / "processed" / "cleaned_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df = df.copy()
    cleaned_df["Loan_Approved"] = y
    cleaned_df.to_csv(output_path, index=False)

    return df, x, y
