import pandas as pd
import hydra
from omegaconf import DictConfig
import joblib
import os

from features import NumericalHandeller, CategoricalHandeller, ImputeStrategy, Encoder, EncodingStrategy

@hydra.main(config_path="../../", config_name="config", version_base="1.2")
def data_preprocessing(cfg: DictConfig):
    train_df = pd.read_csv("data/interim/train_raw.csv") 
    test_df = pd.read_csv("data/interim/test_raw.csv")

    TARGET = cfg.params.target
    
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    age_handler = NumericalHandeller(column="Age", impute_strategy=ImputeStrategy.MEAN)
    fare_handler = NumericalHandeller(column="Fare", impute_strategy=ImputeStrategy.MEDIAN)
    sex_handler = CategoricalHandeller(column="Sex")
    embarked_handler = CategoricalHandeller(column="Embarked")

    handlers = [age_handler, fare_handler, sex_handler, embarked_handler]
    for handler in handlers:
        handler.fit(X_train)
        X_train = handler.transform(X_train)
        X_test = handler.transform(X_test)

    sex_encoder = Encoder("Sex", EncodingStrategy.LABEL)
    embarked_encoder = Encoder("Embarked", EncodingStrategy.ONE_HOT)

    sex_encoder.fit(X_train)
    embarked_encoder.fit(X_train)

    X_train = sex_encoder.transform(X_train)
    X_test = sex_encoder.transform(X_test)
    X_train = embarked_encoder.transform(X_train)
    X_test = embarked_encoder.transform(X_test)

    os.makedirs("data/processed", exist_ok=True)
    
    train_processed = pd.concat([X_train, y_train], axis=1)
    test_processed = pd.concat([X_test, y_test], axis=1)
    
    train_processed.to_csv("data/processed/train_processed.csv", index=False)
    test_processed.to_csv("data/processed/test_processed.csv", index=False)

    artifacts = {
        "age_handler": age_handler,
        "fare_handler": fare_handler,
        "sex_handler": sex_handler,
        "embarked_handler": embarked_handler,
        "sex_encoder": sex_encoder,
        "embarked_encoder": embarked_encoder
    }
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(artifacts, "models/preprocessor.joblib")
    
    print(" Preprocessing completed and artifacts saved via Hydra!")

if __name__ == "__main__":
    data_preprocessing()