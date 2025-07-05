from sklearn.model_selection import train_test_split
import pandas as pd
import os

os.getcwd()

# define paths 
X_train_path = r"C:\Users\Jessica J. Ugowe\Documents\HERBST 2024\Python for healthcare\Diabetes_Prediction_Project\diabetes_prediction\data\processed\X_train_imputed.csv"
y_train_path = r"C:\Users\Jessica J. Ugowe\Documents\HERBST 2024\Python for healthcare\Diabetes_Prediction_Project\diabetes_prediction\data\processed\y_train.csv"

# Load the training dataset
X_train_full = pd.read_csv(X_train_path)
y_train_full = pd.read_csv(y_train_path)

print(y_train_full.value_counts())
y_train_full = y_train_full['Diabetes_binary']  

# Split into Training and Calibration
X_train, X_calibration, y_train, y_calibration = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# Save the split datasets
X_train.to_csv(r"C:\Users\Jessica J. Ugowe\Documents\HERBST 2024\Python for healthcare\Diabetes_Prediction_Project\diabetes_prediction\data\processed\X_train_split.csv", index = False)
y_train.to_csv(r"C:\Users\Jessica J. Ugowe\Documents\HERBST 2024\Python for healthcare\Diabetes_Prediction_Project\diabetes_prediction\data\processed\y_train_split.csv", index = False)
X_calibration.to_csv(r"C:\Users\Jessica J. Ugowe\Documents\HERBST 2024\Python for healthcare\Diabetes_Prediction_Project\diabetes_prediction\data\processed\X_calibration.csv", index = False)
y_calibration.to_csv(r"C:\Users\Jessica J. Ugowe\Documents\HERBST 2024\Python for healthcare\Diabetes_Prediction_Project\diabetes_prediction\data\processed\y_calibration.csv", index = False)

# Print verification
print("Training dataset split into 80% training and 20% calibration!")
print("\nVerifying saved files:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_calibration shape: {X_calibration.shape}")
print(f"y_calibration shape: {y_calibration.shape}")