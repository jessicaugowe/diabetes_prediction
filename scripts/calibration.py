import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score
from joblib import load, dump
from sklearn.preprocessing import StandardScaler

# Paths to train and calibration dataset
X_train_path = r"C:\Users\Jessica J. Ugowe\Documents\HERBST 2024\Python for healthcare\Diabetes_Prediction_Project\diabetes_prediction\data\processed\X_train_split.csv"

X_calibrate_path = r"C:\Users\Jessica J. Ugowe\Documents\HERBST 2024\Python for healthcare\Diabetes_Prediction_Project\diabetes_prediction\data\processed\X_calibration.csv"
y_calibrate_path = r"C:\Users\Jessica J. Ugowe\Documents\HERBST 2024\Python for healthcare\Diabetes_Prediction_Project\diabetes_prediction\data\processed\y_calibration.csv"
# Paths to the validation dataset
X_val_path = r"C:\Users\Jessica J. Ugowe\Documents\HERBST 2024\Python for healthcare\Diabetes_Prediction_Project\diabetes_prediction\data\processed\X_val_log.csv"
y_val_path = r"C:\Users\Jessica J. Ugowe\Documents\HERBST 2024\Python for healthcare\Diabetes_Prediction_Project\diabetes_prediction\data\processed\y_val.csv"
# Load calibration data
X_train = pd.read_csv(X_train_path)
X_cal = pd.read_csv(X_calibrate_path)
y_cal = pd.read_csv(y_calibrate_path)
X_val = pd.read_csv(X_val_path)
y_val = pd.read_csv(y_val_path)


# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
# Load saved models
decision_tree_model = load("decision_tree_model.pkl")
xgboost_model = load("xgboost_model.pkl")

# Fit Scaler on the train dataset Jessica
# Scale calibration data because scaling was applied during training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cal_scaled = scaler.transform(X_cal)

X_cal_scaled_df = pd.DataFrame(X_cal_scaled, columns= X_cal.columns) # Convert scaled data to a dataframe

# Calibrate Decision Tree
dt_calibrator = CalibratedClassifierCV(estimator=decision_tree_model, method='isotonic', cv='prefit')
dt_calibrator.fit(X_cal_scaled_df, y_cal.values.ravel())
dump(dt_calibrator, "calibrated_decision_tree_model.pkl")
print("Calibrated Decision Tree model saved as 'calibrated_decision_tree_model.pkl'")

# Calibrate XGBoost
xgb_calibrator = CalibratedClassifierCV(estimator=xgboost_model, method='sigmoid', cv='prefit')
xgb_calibrator.fit(X_cal_scaled_df, y_cal.values.ravel())
dump(xgb_calibrator, "calibrated_xgboost_model.pkl")
print("Calibrated XGBoost model saved as 'calibrated_xgboost_model.pkl'")

# Evaluate calibrated models 
# Evaluate Decision Tree
y_cal_dt_cal_proba = dt_calibrator.predict_proba(X_cal_scaled_df)[:, 1]
# Optimize threshold for calibration
thresholds = np.arange(0.001, 0.9, 0.001)
best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    y_pred_dt = (y_cal_dt_cal_proba >= threshold).astype(int)
    current_f1 = f1_score(y_cal, y_pred_dt)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = threshold

print(f"Optimal Threshold after Calibration: {best_threshold}")
print(f"Best F1 Score after Calibration: {best_f1}")

y_cal_dt_cal_pred = (y_cal_dt_cal_proba >= best_threshold).astype(int)
print(classification_report(y_cal, y_cal_dt_cal_pred))
print(f"F1 Score: {f1_score(y_cal, y_cal_dt_cal_pred):.2f}")


# Evaluate XGBoost
# Evaluate XGBoost on Validation Dataset
y_val_xgb_cal_proba = xgb_calibrator.predict_proba(X_val_scaled)[:, 1]

# Optimize threshold for validation dataset
thresholds = np.arange(0.001, 0.99, 0.001)
best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    y_pred_xgb = (y_val_xgb_cal_proba >= threshold).astype(int)
    current_f1 = f1_score(y_val, y_pred_xgb)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = threshold

print(f"Optimal Threshold after Calibration: {best_threshold}")
print(f"Best F1 Score after Calibration: {best_f1}")

# Final predictions on validation dataset
y_val_xgb_cal_pred = (y_val_xgb_cal_proba >= best_threshold).astype(int)

# Evaluate performance on validation dataset
print(classification_report(y_val, y_val_xgb_cal_pred))
print(f"F1 Score: {f1_score(y_val, y_val_xgb_cal_pred):.2f}")

# Save calibrated XGBoost threshold
dump(best_threshold, "calibrated_xgboost_model_threshold.pkl")
print("Calibrated XGBoost model threshold saved as 'calibrated_xgboost_model_threshold.pkl'")


# Save calibrated XGBoost threshold
dump(best_threshold, "calibrated_xgboost_model_threshold.pkl")
print("Calibrated XGBoost model threshold saved as 'calibrated_xgboost_model_threshold.pkl'")