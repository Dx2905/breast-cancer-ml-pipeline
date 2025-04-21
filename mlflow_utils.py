import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


    # Log to MLflow
def log_confusion_matrix(cm, model_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    os.makedirs("artifacts", exist_ok=True)
    path = f"artifacts/{model_name}_confusion_matrix.png"
    plt.savefig(path)
    plt.close()

    mlflow.log_artifact(path)

def log_model_metrics(name, model, X_test, y_test, metrics_dict, params=None):
    with mlflow.start_run(run_name=name):
        y_pred = model.predict(X_test)

        # Log model with signature and input example
        signature = infer_signature(X_test, y_pred)
        input_example = X_test[:1]
        mlflow.sklearn.log_model(model, name, signature=signature, input_example=input_example)

        # Log scalar metrics
        for metric_name, value in metrics_dict.items():
            try:
                mlflow.log_metric(metric_name, float(value))
            except Exception as e:
                print(f"Failed to log metric '{metric_name}': {e}")

        # Log confusion matrix as artifact
        if "Confusion Matrix" in metrics_dict:
            log_confusion_matrix(metrics_dict["Confusion Matrix"], name)

        # Log hyperparameters
        if params:
            mlflow.log_params(params)

