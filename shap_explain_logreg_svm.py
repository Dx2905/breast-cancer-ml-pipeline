import os
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# === Setup ===
os.makedirs("plots", exist_ok=True)

# === Load Data ===
data = pd.read_csv("wdbc.csv")
data.columns = [
    'id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
    'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

# Encode label
data['diagnosis'] = LabelEncoder().fit_transform(data['diagnosis'])

X = data.iloc[:, 2:]
y = data['diagnosis']
feature_names = X.columns

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Explainability for LOGISTIC REGRESSION ===
print("\n[LogReg] Computing SHAP values...")
logreg_model = joblib.load("models/logreg.pkl")
logreg_explainer = shap.Explainer(logreg_model, X_scaled)
logreg_shap_values = logreg_explainer(X_scaled)

shap.summary_plot(logreg_shap_values, X_scaled, feature_names=feature_names, show=False)
plt.savefig("plots/shap_summary_logreg.png")
plt.clf()

shap.summary_plot(logreg_shap_values, X_scaled, feature_names=feature_names, plot_type="bar", show=False)
plt.savefig("plots/shap_bar_logreg.png")
plt.clf()

# === Explainability for SVM LINEAR ===
print("\n[SVM Linear] Computing SHAP values...")
svm_model = joblib.load("models/svm_linear.pkl")
svm_explainer = shap.Explainer(svm_model, X_scaled)
svm_shap_values = svm_explainer(X_scaled)

shap.summary_plot(svm_shap_values, X_scaled, feature_names=feature_names, show=False)
plt.savefig("plots/shap_summary_svm.png")
plt.clf()

shap.summary_plot(svm_shap_values, X_scaled, feature_names=feature_names, plot_type="bar", show=False)
plt.savefig("plots/shap_bar_svm.png")
plt.clf()

print("\n✅ SHAP explainability completed for Logistic Regression and SVM. Check the 'plots/' folder.")
