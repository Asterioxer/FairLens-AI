
# src/bias_detection.py
from fairlearn.metrics import demographic_parity_difference
import pandas as pd

def detect_bias(y_pred, sensitive_data, sensitive_attribute='sex'):
    # Ensure sensitive_data is aligned with y_pred
    dpd = demographic_parity_difference(y_true=None,  # Only predictions needed
                                       y_pred=y_pred,
                                       sensitive_features=sensitive_data[sensitive_attribute])
    return dpd

# Test in Colab
from train_model import train_model
from data_preprocessing import load_and_preprocess_data

X, y, sensitive_data = load_and_preprocess_data()
_, X_test, _, y_pred = train_model(X, y)
# Align sensitive_data with test set
sensitive_data_test = sensitive_data.loc[X_test.index]
dpd = detect_bias(y_pred, sensitive_data_test)
print(f"Demographic Parity Difference (sex): {dpd:.3f}")
