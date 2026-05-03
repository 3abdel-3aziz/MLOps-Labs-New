import json
import os
import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig
from utils.logger import get_logger 

import mlflow.xgboost  
import dagshub
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

logger = get_logger(__name__)

dagshub.init(repo_owner='ahmed2025.mohamed2000', repo_name='MLOps-Labs-New', mlflow=True)

@hydra.main(config_path="../../", config_name="config", version_base="1.2")
def evaluate(cfg: DictConfig):
    try:
        logger.info(">>> Starting Model Evaluation Stage <<<")
        
        with mlflow.start_run():
            test_df = pd.read_csv("data/processed/test_processed.csv")
            model = joblib.load("models/model.joblib")
            logger.info(f"Test data and model loaded successfully. Test shape: {test_df.shape}")

            TARGET = cfg.params.target
            X_test = test_df.drop(columns=[TARGET])
            y_test = test_df[TARGET]

            logger.info("Executing model predictions on test set...")
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds)
            
            logger.info(f"Evaluation Metrics Calculated - Accuracy: {acc:.4f}")
            logger.info(f"\nDetailed Classification Report:\n{report}")

            os.makedirs("reports", exist_ok=True)
            metrics = {
                "accuracy": round(acc, 6),
                "f1_weighted": round(f1_score(y_test, preds, average="weighted"), 6),
                "precision_weighted": round(precision_score(y_test, preds, average="weighted"), 6),
                "recall_weighted": round(recall_score(y_test, preds, average="weighted"), 6),
            }
            
            with open("reports/metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            logger.info("Metrics saved to reports/metrics.json")

            with open("reports/metrics.txt", "w") as f:
                f.write(f"Accuracy: {acc}\n")
                f.write(report)
            
            mlflow.log_metrics(metrics)
            logger.info("Metrics successfully logged to MLflow/DagsHub.")

            logger.info("Model Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error occurred during Model Evaluation: {str(e)}")
        raise e

if __name__ == "__main__":
    evaluate()