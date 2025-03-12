# **Credit Risk Prediction Model**

## **Project Overview**  
The Credit Risk Prediction Model helps financial institutions assess the probability of loan defaults using machine learning. It predicts the risk level of loan applicants based on their financial history and provides a credit score and rating.

## **Features**  
- Predicts default probability based on user input  
- Computes credit score and assigns a risk rating  
- Supports multiple loan types and purposes  

## **Tech Stack**  
- **Programming Language:** Python  
- **Frameworks:** Streamlit (for UI)  
- **Libraries:**  
  - **Machine Learning:** scikit-learn, joblib  
  - **Data Processing:** Pandas, NumPy  
  - **Model Deployment:** Streamlit  

## **Usage**  
1. Open the **[Credit Risk Prediction UI](https://predict-credit-risk.streamlit.app/)**  
2. Enter the required financial details  
3. Click **Predict Risk** to get:  
   - **Default Probability**  
   - **Credit Score**  
   - **Credit Rating**  

## **Model Details**  
- Uses **logistic regression**  
- **Hyperparameters used:**  
  - **C:** 2.28  
  - **Solver:** liblinear  
  - **Tolerance (tol):** 0.0000026  
  - **Class weight:** None  
- Features include **loan amount, income, credit utilization ratio, age, etc.**  
- Computes **loan-to-income ratio, delinquency rate, and average DPD**  

## **Evaluation Metrics**  
- **Recall:** 94%  
- **Accuracy:** 93%  
- **AUC:** 0.98  
- **KS Statistic:** 85% in the first three deciles  

## **Future Enhancements**  
- **Deep learning-based risk analysis**
