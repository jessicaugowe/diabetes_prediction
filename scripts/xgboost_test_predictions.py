
import pandas as pd
import numpy as np
from joblib import load

# Paths to test dataset
X_test_path = r"C:\Users\Jessica J. Ugowe\Documents\HERBST 2024\Python for healthcare\Diabetes_Prediction_Project\diabetes_prediction\data\processed\X_test_log.csv"

# Load calibrated XGBoost model
xgboost_model = load("calibrated_xgboost_model.pkl")

# Load scaler fitted on 80% training data
scaler_80 = load("scaler_80.pkl")

# Load test dataset
X_test = pd.read_csv(X_test_path)

# Scale test dataset
X_test_scaled = scaler_80.transform(X_test)

# Predict probabilities for the test dataset
y_test_proba = xgboost_model.predict_proba(X_test_scaled)[:, 1]

# Use optimized threshold from the validation set
optimal_threshold = load("calibrated_xgboost_model_threshold.pkl")
# Generate final predictions
final_test_predictions = (y_test_proba >= optimal_threshold).astype(int)

# Save predictions to a CSV file
pd.DataFrame({
    "Index": np.arange(len(final_test_predictions)),  
    "Diabetes_binary": final_test_predictions
}).to_csv("diabetes-labels-test.csv", index=False)

print("Final XGBoost predictions saved to 'diabetes-labels-test.csv'.")
