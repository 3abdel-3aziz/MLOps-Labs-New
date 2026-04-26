import mlflow
import pandas as pd
import joblib

dagshub_url = "https://dagshub.com/ahmed2025.mohamed2000/MLOps-Labs-New.mlflow"
mlflow.set_tracking_uri(dagshub_url)

MODEL_NAME = "Titanic_XGBoost_Model"
MODEL_VERSION = "1" 

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

print(f"--- Connecting to DagsHub Registry: {dagshub_url} ---")

try:
    model = mlflow.pyfunc.load_model(model_uri)
    print("Success! Model loaded from DagsHub.")

    sample_input = pd.DataFrame([{
            "Pclass": 3,
            "Sex": 1,
            "Age": 25,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 7.25,
            "Embarked_C": 0,
            "Embarked_Q": 0,
            "Embarked_S": 1
        }])

    prediction = model.predict(sample_input)
    result = "Survived" if prediction[0] == 1 else "Not Survived"
    print(f"\nResult: {result}")

except Exception as e:
    print(f" Error: {e}")
    print("\n--- Emergency Fallback (Loading Locally) ---")
    model = joblib.load("models/model.joblib")
    print(" Loaded from local 'models/model.joblib' instead.")
    prediction = model.predict(sample_input)
    print(f"Result (Local): {'Survived' if prediction[0] == 1 else 'Not Survived'}")