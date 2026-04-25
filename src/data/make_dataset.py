import pandas as pd
from abc import ABC, abstractmethod
import os
class DataIngestion(ABC):
    @abstractmethod
    def load_data(self):
        pass    

class CSVDataIngestion(DataIngestion):
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        return pd.read_csv(self.file_path) 

class DataSplitter:
    def split_data(self, df, target_col, test_size, random_state):
        from sklearn.model_selection import train_test_split
        x = df.drop(target_col, axis=1)
        y = df[target_col]
        return train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    def save_data(self, train_df, test_df, output_path):
        os.makedirs(output_path, exist_ok=True)
        train_df.to_csv(os.path.join(output_path, "train_raw.csv"), index=False)
        test_df.to_csv(os.path.join(output_path, "test_raw.csv"), index=False)
        print(f"Raw data split and saved to {output_path}")