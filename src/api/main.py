from fastapi import FastAPI, HTTPException
import uvicorn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.api.schemas import TitanicInput, TitanicOutput, TitanicBatchInput, TitanicBatchOutput
from src.api.model_handler import ModelHandler

app = FastAPI(title="Titanic Survival Predictor")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib") 
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")

try:
    handler = ModelHandler(model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH)
    print(" API is ready and model is loaded!")
except Exception as e:
    print(f" Failed to load model assets: {e}")
    handler = None

@app.post("/predict", response_model=TitanicOutput)
def predict_survival(payload: TitanicInput):
    if not handler:
        raise HTTPException(status_code=500, detail="Model assets not found.")
    
    try:
        prediction, probability = handler.predict(payload)
        
        return TitanicOutput(
            prediction=prediction,
            probability=probability,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch", response_model=TitanicBatchOutput)
def predict_batch(payload: TitanicBatchInput):
    if not handler:
        raise HTTPException(status_code=500, detail="Server model is not initialized.")
    
    batch_results = []
    try:
        for item in payload.inputs:
            pred, prob = handler.predict(item)
            batch_results.append(TitanicOutput(prediction=pred, probability=prob))
        
        return TitanicBatchOutput(results=batch_results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)