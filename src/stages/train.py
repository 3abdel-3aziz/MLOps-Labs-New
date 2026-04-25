import pandas as pd
import hydra
from omegaconf import DictConfig
from xgboost import XGBClassifier
import joblib
import os

@hydra.main(config_path="../../", config_name="config", version_base="1.2")
def train(cfg: DictConfig):
    train_df = pd.read_csv("data/processed/train_processed.csv")
    
    TARGET = cfg.params.target
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    model = XGBClassifier(
        n_estimators=cfg.model_params.n_estimators,
        learning_rate=cfg.model_params.learning_rate,
        max_depth=cfg.model_params.max_depth,
        subsample=cfg.model_params.subsample,
        colsample_bytree=cfg.model_params.colsample_bytree,
        random_state=cfg.params.random_state,
        eval_metric='logloss'
    )
    
    print("Training the model via Hydra parameters...")
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    model_path = "models/model.joblib" 
    joblib.dump(model, model_path)
    
    print(f" Model trained and saved to: {model_path}")

if __name__ == "__main__":
    train()