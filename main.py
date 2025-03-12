import streamlit as st
import numpy as np
from prediction_helper import predict_credit_risk


def main():
    st.title("Credit Risk Management ML Model")
    st.write("Enter the necessary details to assess credit risk.")

    # Define base features (user inputs)
    base_features = ["loan_amount", "income", "delinquent_months", "total_loan_months", "total_dpd",
                     "number_of_open_accounts", "credit_utilization_ratio", "age", "loan_tenure_months"]

    # Define categorical features (user selects values)
    categorical_features = {
        "residence_type": ["Mortgage", "Owned", "Rented"],
        "loan_purpose": ["Home", "Personal", "Auto", "Education"],
        "loan_type": ["Secured", "Unsecured"]
    }

    # Create a 5-row, 3-column layout
    row1 = st.columns(3)
    row2 = st.columns(3)
    row3 = st.columns(3)
    row4 = st.columns(3)
    row5 = st.columns(3)

    user_inputs = {}

    # Collect user inputs for base features
    rows = [row1, row2, row3]
    feature_idx = 0

    for row in rows:
        for col in row:
            if feature_idx < len(base_features):
                user_inputs[base_features[feature_idx]] = float(
                    col.text_input(f"{base_features[feature_idx]}", value="0"))
                feature_idx += 1

    # Collect user inputs for categorical features
    row4_idx = 0
    for feature, options in categorical_features.items():
        user_inputs[feature] = row4[row4_idx].selectbox(f"{feature}", options)
        row4_idx += 1

    # Calculate derived features
    if st.button("Predict Risk"):
        user_inputs["loan_to_income"] = round(user_inputs["loan_amount"] / user_inputs["income"], 2) if user_inputs[
                                                                                                            "income"] != 0 else 0
        user_inputs["deliquent_to_total_months"] = round(
            (user_inputs["delinquent_months"] / user_inputs["total_loan_months"]) * 100, 2) if user_inputs[
                                                                                                   "total_loan_months"] != 0 else 0
        user_inputs["avg_dpd_per_deliquency"] = round(user_inputs["total_dpd"] / user_inputs["delinquent_months"], 2) if \
        user_inputs["delinquent_months"] != 0 else 0

        # Call the prediction function
        default_probability,credit_score,rating = predict_credit_risk(user_inputs)

        # Display results
        st.write("loan_to_income ratio:", user_inputs['loan_to_income'])
        st.write("deliquent_to_total_months ratio:", user_inputs['deliquent_to_total_months'])
        st.write("avg_dpd_per_deliquency ratio:", user_inputs['avg_dpd_per_deliquency'])
        st.success(f"default_probability: {round(default_probability*100,2)}")
        st.success(f"credit_score: {credit_score}")
        st.success(f"rating: {rating}")


if __name__ == "__main__":
    main()
