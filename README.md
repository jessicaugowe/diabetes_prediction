
# Machine Learning Model Calibration Project for Diabetes Prediction

This project focuses on training and calibrating multiple classification models to improve probability estimates. It includes data preprocessing, model training and calibration.

---

## Project Structure

```plaintext

├── models/                      # Saved trained and calibrated models
│   ├── decision_tree_model.pkl
│   └── xgboost_model.pkl
├── notebook/                    # Jupyter notebooks for EDA and preprocessing
│   ├── 01_data_exploration.ipynb
│   └── 02_data_imputation.ipynb
├── scripts/                     # Python scripts for training and calibration
│   ├── calibration.py
│   ├── decision_tree_model.py
│   ├── split_data.py
│   ├── xgboost_model.py
│   └── xgboost_test_predictions.py
