

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from joblib import dump

# Paths to the datasets
X_train_path = r"C:\Users\Jessica J. Ugowe\Documents\HERBST 2024\Python for healthcare\Diabetes_Prediction_Project\diabetes_prediction\data\processed\X_train_split.csv"
y_train_path = r"C:\Users\Jessica J. Ugowe\Documents\HERBST 2024\Python for healthcare\Diabetes_Prediction_Project\diabetes_prediction\data\processed\y_train_split.csv"
# Paths to the validation dataset
X_val_path = r"C:\Users\Jessica J. Ugowe\Documents\HERBST 2024\Python for healthcare\Diabetes_Prediction_Project\diabetes_prediction\data\processed\X_val_log.csv"
y_val_path = r"C:\Users\Jessica J. Ugowe\Documents\HERBST 2024\Python for healthcare\Diabetes_Prediction_Project\diabetes_prediction\data\processed\y_val.csv"

# Load datasets
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)
X_val = pd.read_csv(X_val_path)
y_val = pd.read_csv(y_val_path)


# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# Create XGBoost classifier 
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    eval_metric='logloss'
)

# Calculate the scale_pos_weight
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# parameters for RandomizedSearchCV
param_distribution = {
    'max_depth': [3],
    'learning_rate': [ 0.1, 0.2],
    'n_estimators': [500, 1000],
    'min_child_weight': [2, 4],
    'gamma': [2.0, 4.0],
    'subsample': [0.6, 0.7],
    'colsample_bytree': [0.7],
    'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 1.5, scale_pos_weight * 2],
    'reg_alpha': [2, 4],      # L1 regularization term on weights
    'reg_lambda': [2, 4, 6]     #L2 regularization 
 
}


# Random Search
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distribution,
    scoring='f1',
    refit=True,  
    cv=5,
    verbose=2,
    n_jobs=-1,
    n_iter=100
)

# Fit model 
random_search.fit(X_train_scaled, y_train)

# Get the best model and parameters
best_xgb_model = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)
print("Best F1 Score during CV:", random_search.best_score_)

# Save the model
model_filename = "xgboost_model.pkl"
dump(best_xgb_model, model_filename)
print(f"Model saved as {model_filename}")

# Predict probabilities on validation set
y_val_proba = best_xgb_model.predict_proba(X_val_scaled)[:, 1]

# Find optimal threshold for F1 score
thresholds = np.arange(0.001, 0.95, 0.001)
best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    y_pred = (y_val_proba >= threshold).astype(int)
    current_f1 = f1_score(y_val, y_pred)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = threshold

print(f"Optimal threshold: {best_threshold:.2f}")

# Final predictions with optimal threshold
y_val_pred_final = (y_val_proba >= best_threshold).astype(int)

# Print final classification results
print(classification_report(y_val, y_val_pred_final))
print(f"F1 Score: {f1_score(y_val, y_val_pred_final):.2f}")

