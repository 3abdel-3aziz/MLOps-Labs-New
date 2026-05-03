from fastapi import FastAPI, HTTPException
import uvicorn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.api.schemas import TitanicInput, TitanicOutput, TitanicBatchInput, TitanicBatchOutput
from src.api.model_handler import ModelHandler
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Titanic Survival Predictor")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib") 
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")

try:
    logger.info(">>> Initializing API and Loading Model Assets <<<")
    handler = ModelHandler(model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH)
    logger.info("API is ready and model is loaded successfully!")
except Exception as e:
    logger.error(f"Critical Error: Failed to load model assets: {str(e)}")
    handler = None

@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": handler is not None}

@app.post("/predict", response_model=TitanicOutput)
def predict_survival(payload: TitanicInput):
    if not handler:
        logger.error("Prediction attempt failed: Model handler is not initialized.")
        raise HTTPException(status_code=500, detail="Model assets not found.")
    
    try:
        logger.info(f"Received prediction request for: {payload.dict()}")
        
        prediction, probability = handler.predict(payload)
        
        logger.info(f"Prediction result: {prediction} with probability {probability:.4f}")
        
        return TitanicOutput(
            prediction=prediction,
            probability=probability,
            status="success"
        )
    except Exception as e:
        logger.error(f"Prediction error for payload {payload}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch", response_model=TitanicBatchOutput)
def predict_batch(payload: TitanicBatchInput):
    if not handler:
        logger.error("Batch prediction failed: Model handler not initialized.")
        raise HTTPException(status_code=500, detail="Server model is not initialized.")
    
    logger.info(f"Received batch prediction request with {len(payload.inputs)} items.")
    batch_results = []
    try:
        for i, item in enumerate(payload.inputs):
            pred, prob = handler.predict(item)
            batch_results.append(TitanicOutput(prediction=pred, probability=prob))
        
        logger.info(f"Batch prediction completed successfully for {len(payload.inputs)} items.")
        return TitanicBatchOutput(results=batch_results)
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)


