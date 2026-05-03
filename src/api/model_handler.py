import pandas as pd
import joblib
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelHandler:
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self._load_assets()

    def _load_assets(self):
        try:
            logger.info(f"Attempting to load model from: {self.model_path}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            logger.info(f"Attempting to load preprocessor from: {self.preprocessor_path}")
            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")
            
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path) 
            logger.info("Model and Preprocessor assets loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load assets: {str(e)}")
            raise e

    def predict(self, input_data):
        try:
            df = pd.DataFrame([input_data.dict()])
            logger.info(f"New prediction request received. Input: {input_data.dict()}")
            
            df = self.preprocessor["age_handler"].transform(df)
            df = self.preprocessor["fare_handler"].transform(df)
            df = self.preprocessor["sex_handler"].transform(df)
            df = self.preprocessor["embarked_handler"].transform(df)
            
            df = self.preprocessor["sex_encoder"].transform(df)
            df = self.preprocessor["embarked_encoder"].transform(df)
            
            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0].max()
            
            logger.info(f"Prediction successful: Result={prediction}, Probability={probability:.4f}")
            
            return int(prediction), float(probability)
            
        except Exception as e:
            logger.error(f"Prediction failed for input {input_data.dict()}. Error: {str(e)}")
            raise e