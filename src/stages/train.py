import pandas as pd
import hydra
from omegaconf import DictConfig
from xgboost import XGBClassifier
import joblib
import os
import mlflow
import mlflow.xgboost  
import dagshub
from src.utils.logger import get_logger

logger = get_logger(__name__)

dagshub.init(repo_owner='ahmed2025.mohamed2000', repo_name='MLOps-Labs-New', mlflow=True)

@hydra.main(config_path="../../", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    try:
        logger.info(">>> Starting Model Training Stage <<<")
        
        with mlflow.start_run():
            train_df = pd.read_csv("data/processed/train_processed.csv")
            logger.info(f"Loaded processed training data with shape: {train_df.shape}")
            
            TARGET = cfg.params.target
            X_train = train_df.drop(columns=[TARGET])
            y_train = train_df[TARGET]

            logger.info(f"Training with parameters: {cfg.model_params}")
            mlflow.log_params(cfg.model_params)
            mlflow.log_param("target", TARGET)

            model = XGBClassifier(
                n_estimators=cfg.model_params.n_estimators,
                learning_rate=cfg.model_params.learning_rate,
                max_depth=cfg.model_params.max_depth,
                subsample=cfg.model_params.subsample,
                colsample_bytree=cfg.model_params.colsample_bytree,
                random_state=cfg.params.random_state,
                eval_metric='logloss'
            )
            
            logger.info("Fitting XGBClassifier model...")
            model.fit(X_train, y_train)
            logger.info("Model fitting completed successfully.")

            mlflow.xgboost.log_model(model, "xgboost_model")
            logger.info("Model logged to MLflow via DagsHub.")

            os.makedirs("models", exist_ok=True)
            model_path = "models/model.joblib" 
            joblib.dump(model, model_path)
            logger.info(f"Model saved locally at: {model_path}")
            
            mlflow.log_artifact(model_path, artifact_path="saved_model")
            logger.info("Model artifact uploaded to MLflow.")
            
            logger.info("Training Stage completed successfully!")

    except Exception as e:
        logger.error(f"Error during training stage: {str(e)}")
        raise e

if __name__ == "__main__":
    train()