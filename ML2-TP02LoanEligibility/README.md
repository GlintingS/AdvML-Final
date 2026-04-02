# Loan Eligibility Predictor

A machine learning project that predicts loan eligibility using **Logistic Regression** and **Random Forest Classifier** models. Includes a Streamlit web application for interactive predictions.

## Project Structure

```
ML2-TP02LoanEligibility/
├── main.py                              # CLI pipeline: preprocess, train, evaluate, plot
├── streamlit_02LoanEligibility_app.py   # Streamlit web app for interactive predictions
├── verify.py                            # Project structure verification script
├── requirements.txt                     # Python dependencies
├── data/
│   ├── raw/credit.csv                   # Raw loan application dataset
│   └── processed/cleaned_data.csv       # Cleaned/encoded dataset (generated)
├── models/                              # Saved trained models (.pkl)
├── scr/
│   ├── data/make_dataset.py             # Data loading, cleaning, and encoding
│   ├── Model/train_models.py            # Model training (LR + RF)
│   ├── Model/predict_models.py          # Model evaluation metrics
│   └── visuals/visualize.py             # Plotting utilities
└── mae_comparison.png                   # Model comparison chart (generated)
```

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`:
  - **Streamlit** — web application framework
  - **pandas** — data manipulation
  - **scikit-learn** — machine learning models and metrics
  - **matplotlib** — chart generation

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

### CLI Pipeline

Runs the full pipeline — data preprocessing, model training, evaluation, and chart generation:

```bash
python main.py
```

This will:
1. Load and preprocess `data/raw/credit.csv`
2. Train Logistic Regression and Random Forest models
3. Save trained models to `models/`
4. Print classification metrics (accuracy, precision, recall, F1)
5. Generate a model comparison chart (`mae_comparison.png`)
6. Write logs to `pipeline.log`

### Streamlit Web App

Launch the interactive web application:

```bash
streamlit run streamlit_02LoanEligibility_app.py
```

The app allows you to:
- Enter applicant details and get a loan eligibility prediction
- View model evaluation metrics
- Explore feature importance and dataset insights

> **Note:** Run `python main.py` first to generate the trained model files required by the Streamlit app.

## Logging

All modules produce structured log output. The CLI pipeline writes logs to `pipeline.log` and the Streamlit app writes to `streamlit_app.log`.
