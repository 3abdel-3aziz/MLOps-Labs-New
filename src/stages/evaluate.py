import json
import os

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig
import mlflow.xgboost  
import dagshub
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

dagshub.init(repo_owner='ahmed2025.mohamed2000', repo_name='MLOps-Labs-New', mlflow=True)
@hydra.main(config_path="../../", config_name="config", version_base="1.2")
def evaluate(cfg: DictConfig):
    with mlflow.start_run():

        test_df = pd.read_csv("data/processed/test_processed.csv")
        
        model = joblib.load("models/model.joblib")

        TARGET = cfg.params.target
        X_test = test_df.drop(columns=[TARGET])
        y_test = test_df[TARGET]

        print("Evaluating model...")
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)

        print(f"\n Model Accuracy: {acc:.4f}")
        print("\nDetailed Classification Report:")
        print(report)

        os.makedirs("reports", exist_ok=True)

        metrics = {
            "accuracy": round(acc, 6),
            "f1_weighted": round(f1_score(y_test, preds, average="weighted"), 6),
            "precision_weighted": round(precision_score(y_test, preds, average="weighted"), 6),
            "recall_weighted": round(recall_score(y_test, preds, average="weighted"), 6),
        }
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open("reports/metrics.txt", "w") as f:
            f.write(f"Accuracy: {acc}\n")
            f.write(report)
        mlflow.log_metrics(metrics)    

        print(f" Metrics saved -> reports/metrics.json")
        print(" Evaluation completed and metrics saved via Hydra!")

if __name__ == "__main__":
    evaluate()