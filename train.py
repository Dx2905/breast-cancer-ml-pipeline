import os
os.makedirs("models", exist_ok=True)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from scipy.stats import uniform, randint, expon, reciprocal
import joblib
from data_preprocessing import load_and_preprocess_data
from mlflow_utils import log_model_metrics


# Load and preprocess data
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data("wdbc.csv")

# Dictionary to store models and results
models = {}
results = {}
best_params= {}
model_results = {}

# 1. Logistic Regression
logreg_params = {'C': uniform(0.1, 100), 'solver': ['newton-cg', 'lbfgs', 'liblinear'], 'max_iter': randint(100, 2000)}
logreg = RandomizedSearchCV(LogisticRegression(), logreg_params, n_iter=50, cv=10, n_jobs=-1, random_state=0)
logreg.fit(X_train, y_train)
models['logreg'] = logreg.best_estimator_
best_params['logreg'] = logreg.best_params_

# 2. KNN
knn_params = {'n_neighbors': randint(3, 20), 'metric': ['euclidean', 'manhattan', 'minkowski'], 'weights': ['distance']}
knn = RandomizedSearchCV(KNeighborsClassifier(), knn_params, n_iter=50, cv=10, n_jobs=-1, random_state=0)
knn.fit(X_train, y_train)
models['knn'] = knn.best_estimator_
best_params['knn'] = knn.best_params_

# 3. SVM Linear
svm_linear_params = {'C': expon(scale=100), 'kernel': ['linear']}
svm_linear = RandomizedSearchCV(SVC(probability=True), svm_linear_params, n_iter=50, cv=10, n_jobs=-1, random_state=0)
svm_linear.fit(X_train, y_train)
models['svm_linear'] = svm_linear.best_estimator_
best_params['svm_linear'] = svm_linear.best_params_

# 4. SVM RBF
svm_rbf_params = {'C': expon(scale=100), 'gamma': reciprocal(0.01, 1.0)}
svm_rbf = RandomizedSearchCV(SVC(kernel='rbf', probability=True), svm_rbf_params, n_iter=50, cv=10, n_jobs=-1, random_state=0)
svm_rbf.fit(X_train, y_train)
models['svm_rbf'] = svm_rbf.best_estimator_
best_params['svm_rbf'] = svm_rbf.best_params_

# 5. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
models['naive_bayes'] = nb
best_params['naive_bayes'] = {} 

# 6. Decision Tree
dt_params = {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
dt = RandomizedSearchCV(DecisionTreeClassifier(criterion='entropy'), dt_params, n_iter=10, cv=10, n_jobs=-1, random_state=0)
dt.fit(X_train, y_train)
models['decision_tree'] = dt.best_estimator_
best_params['decision_tree'] = dt.best_params_

# 7. Random Forest
rf_params = {'n_estimators': randint(100, 300), 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
rf = RandomizedSearchCV(RandomForestClassifier(criterion='entropy'), rf_params, n_iter=10, cv=10, n_jobs=-1, random_state=0)
rf.fit(X_train, y_train)
models['random_forest'] = rf.best_estimator_
best_params['random_forest'] = rf.best_params_

# Evaluate and store results
for name, model in models.items():
    y_pred = model.predict(X_test)
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }
    # Save model
    joblib.dump(model, f"models/{name}.pkl")

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv("model_results.csv")
print(results_df)

print("Best params keys:", best_params.keys())


from mlflow_utils import log_model_metrics

for name, model in models.items():
    # log_model_metrics(name, model, X_test, Y_test, model_results[name])
    log_model_metrics(name, model, X_test, y_test, results[name], params=best_params[name])
    


