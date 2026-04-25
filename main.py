import os
import sys

sys.path.append(os.getcwd())

from src.stages.data_load import data_load
from src.stages.data_preprocess import data_preprocessing
from src.stages.train import train
from src.stages.evaluate import evaluate

def run_pipeline():
    print(" Starting Titanic MLOps Pipeline...\n")
    
    print("Step 1: Loading Data...")
    data_load()
    
    print("\nStep 2: Preprocessing Data...")
    data_preprocessing()
    
    print("\nStep 3: Training Model...")
    train()
    
    print("\nStep 4: Evaluating Model...")
    evaluate()
    
    print("\n Pipeline executed successfully!")

if __name__ == "__main__":
    run_pipeline()