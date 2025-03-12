import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = 'artifacts/model_data.joblib'

# Load trained model, scaler, and feature information
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']

# Model coefficients & intercept (for probability calculation)
coefficients = model.coef_[0]
intercept = model.intercept_[0]

# Base score & scaling factor for CIBIL score calculation
BASE_SCORE = 300
SCALE_FACTOR = 600

# List of all numeric columns (some might not be in 'features')
all_numeric_columns = [
    'number_of_open_accounts', 'number_of_closed_accounts', 'enquiry_count',
    'credit_utilization_ratio', 'age', 'number_of_dependants',
    'years_at_current_address', 'zipcode', 'sanction_amount',
    'processing_fee', 'gst', 'net_disbursement', 'loan_tenure_months',
    'principal_outstanding', 'bank_balance_at_application',
    'loan_to_income', 'deliquent_to_total_months', 'avg_dpd_per_deliquency'
]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict_credit_risk(user_input):
    # Initialize dictionary with user inputs
    data = {
        'number_of_open_accounts': user_input['number_of_open_accounts'],
        'credit_utilization_ratio': user_input['credit_utilization_ratio'],
        'age': user_input['age'],
        'loan_tenure_months': user_input['loan_tenure_months'],
        'loan_to_income': user_input['loan_to_income'],
        'deliquent_to_total_months': user_input['deliquent_to_total_months'],
        'avg_dpd_per_deliquency': user_input['avg_dpd_per_deliquency'],

        # One-hot encoding for categorical features
        'residence_type_Owned': 1 if user_input['residence_type'] == "Owned" else 0,
        'residence_type_Rented': 1 if user_input['residence_type'] == "Rented" else 0,
        'loan_purpose_Education': 1 if user_input['loan_purpose'] == "Education" else 0,
        'loan_purpose_Home': 1 if user_input['loan_purpose'] == "Home" else 0,
        'loan_purpose_Personal': 1 if user_input['loan_purpose'] == "Personal" else 0,
        'loan_type_Unsecured': 1 if user_input['loan_type'] == "Unsecured" else 0
    }

    # Add dummy values for missing numerical columns
    for col in all_numeric_columns:
        if col not in data:
            data[col] = 0

            # Convert dictionary to DataFrame
    df = pd.DataFrame([data])

    # Scale numerical features (only those present in the model)
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Ensure feature order matches the trained model
    df = df[features]

    # Compute linear combination (logits)
    logit = np.dot(df.iloc[0], coefficients) + intercept

    # Apply sigmoid function to get default probability
    default_probability = sigmoid(logit)

    # Non-default probability
    non_default_probability = 1 - default_probability

    # Compute CIBIL score
    cibil_score = round(BASE_SCORE + (non_default_probability * SCALE_FACTOR), 2)

    # Assign credit rating
    if 750 <= cibil_score <= 900:
        rating = "Excellent"
    elif 650 <= cibil_score < 750:
        rating = "Good"
    elif 500 <= cibil_score < 650:
        rating = "Average"
    elif 300 <= cibil_score < 500:
        rating = "Poor"
    else:
        rating = "Undefined"

    return default_probability,cibil_score,rating

    # return {
    #
    #     "probability_of_default": round(default_probability, 4),
    #     "non_default_probability": round(non_default_probability, 4),
    #     "cibil_score": cibil_score,
    #     "rating": rating
    # }
