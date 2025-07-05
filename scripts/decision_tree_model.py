
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Save the fitted scaler for decision tree and XGBoost...Next time do this at the beginning Jessica
dump(scaler, "scaler_80.pkl")

X_val_scaled = scaler.transform(X_val)

# Decision Tree hyperparameter grid
dt_param_grid = {
    'criterion': ['gini'],  # Splitting criteria
    'max_depth': [9],  # Depth of the tree
    'min_samples_split': [500],  # Minimum samples to split a node
    'min_samples_leaf': [500],  # Minimum samples in a leaf
    'class_weight': ['balanced']  # To Handle class imbalance
}
# Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Grid Search for hyperparameter tuning
dt_grid_search = GridSearchCV(
    estimator=dt_model,
    param_grid=dt_param_grid,
    scoring='f1',  # Optimise for F1 Score
    cv=5,
    verbose=2,
    n_jobs=-1
)

# Fit model
dt_grid_search.fit(X_train, y_train.values.ravel())

# Get the best model
best_dt_model = dt_grid_search.best_estimator_
print("Best Parameters for Decision Tree:", dt_grid_search.best_params_)

# Save the best Decision Tree model
dump(best_dt_model, "decision_tree_model.pkl")
print("Decision Tree model saved as 'decision_tree_model.pkl'.")

# Predict probabilities for threshold tuning
y_val_dt_proba = best_dt_model.predict_proba(X_val)[:, 1]

# Find optimal threshold
thresholds = np.arange(0.1, 0.9, 0.01)
best_threshold_dt = 0.5
best_f1_dt = 0

for threshold in thresholds:
    y_pred = (y_val_dt_proba >= threshold).astype(int)
    current_f1 = f1_score(y_val, y_pred)
    if current_f1 > best_f1_dt:
        best_f1_dt = current_f1
        best_threshold_dt = threshold

# Final predictions with the best threshold
y_val_pred_dt_final = (y_val_dt_proba >= best_threshold_dt).astype(int)

# Print results
print(f"\nOptimal Threshold for Decision Tree: {best_threshold_dt:.2f}")
print("\nClassification Report for Decision Tree:")
print(classification_report(y_val, y_val_pred_dt_final))
print(f"\nF1 Score for Decision Tree: {best_f1_dt:.2f}")
