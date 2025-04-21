import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# === Load Data ===
data = pd.read_csv("wdbc.csv")
data.columns = [
    'id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
    'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

# Encode target
data['diagnosis'] = LabelEncoder().fit_transform(data['diagnosis'])

# Separate features and target
X = data.iloc[:, 2:]
y = data['diagnosis']
feature_names = X.columns

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_df = pd.DataFrame(X_scaled, columns=feature_names)

# === Load Logistic Regression model ===
model = joblib.load("models/logreg.pkl")

# === SHAP Explainability ===
print("Computing SHAP values...")
explainer = shap.Explainer(model, X_df)
shap_values = explainer(X_df)

# === Plot SHAP Summary ===
print("Saving SHAP summary plots...")

# Beeswarm plot
shap.summary_plot(shap_values.values, X_df, show=True)

# Bar plot
shap.summary_plot(shap_values.values, X_df, plot_type="bar", show=True)
