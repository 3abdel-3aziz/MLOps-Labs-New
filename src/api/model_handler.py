import pandas as pd
import joblib
import os

class ModelHandler:
    def __init__(self, model_path:str , preprocessor_path :str) :
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self._load_assets()

    def _load_assets(self):
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")
            
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path) 
            print(" Model and Preprocessor loaded successfully")
    def predict(self, input_data):
            df = pd.DataFrame([input_data.dict()])
            
            
            df = self.preprocessor["age_handler"].transform(df)
            df = self.preprocessor["fare_handler"].transform(df)
            df = self.preprocessor["sex_handler"].transform(df)
            df = self.preprocessor["embarked_handler"].transform(df)
            
            df = self.preprocessor["sex_encoder"].transform(df)
            df = self.preprocessor["embarked_encoder"].transform(df)
            
            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0].max()
            
            return int(prediction), float(probability)

