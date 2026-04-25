import pandas as pd
import numpy as np 
from abc import ABC, abstractmethod
from enum import Enum
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os 
class ImputeStrategy(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"

class Handeller(ABC):
    @abstractmethod
    def fit(self, df):
        pass

    def log_scale(self, df, column):
        df_copy = df.copy()
        df_copy[column] = np.log(df_copy[column] + 1)
        return df_copy

    def drop_column(self, df, column: list):
        for col in column:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in dataframe")            
        return df.drop(columns=column)

class NumericalHandeller(Handeller):
    def __init__(self, column, impute_strategy: ImputeStrategy, group_by_column=None):
        self.column = column 
        self.group_by_column = group_by_column
        self.impute_strategy = impute_strategy
        self.learned_number = None

    def fit(self, df):
        if self.group_by_column:
            if self.impute_strategy == ImputeStrategy.MEAN:
                self.learned_number = df.groupby(self.group_by_column)[self.column].mean()
            elif self.impute_strategy == ImputeStrategy.MEDIAN:
                self.learned_number = df.groupby(self.group_by_column)[self.column].median()
        else:
            if self.impute_strategy == ImputeStrategy.MEAN:
                self.learned_number = df[self.column].mean()
            elif self.impute_strategy == ImputeStrategy.MEDIAN:
                self.learned_number = df[self.column].median()
        return self

    def transform(self, df):
        df_copy = df.copy()
        if self.group_by_column and isinstance(self.learned_number, pd.Series):
            df_copy[self.column] = df_copy[self.column].fillna(df_copy[self.group_by_column].map(self.learned_number))
        else:
            df_copy[self.column] = df_copy[self.column].fillna(self.learned_number)
        return df_copy

class CategoricalHandeller(Handeller): 
    def __init__(self, column, group_by_column=None):
        self.column = column 
        self.group_by_column = group_by_column
        self.learned_number = None

    def fit(self, df):
        if self.group_by_column is not None:
            self.learned_number = df.groupby(self.group_by_column)[self.column].apply(
                lambda x: x.mode()[0] if not x.mode().empty else None
            )
        else:
            self.learned_number = df[self.column].mode()[0]
        return self

    def transform(self, df):
        df_copy = df.copy()
        if self.group_by_column is not None:
            df_copy[self.column] = df_copy[self.column].fillna(df_copy[self.group_by_column].map(self.learned_number))
        else:
            df_copy[self.column] = df_copy[self.column].fillna(self.learned_number)
        return df_copy

class EncodingStrategy(Enum):
    ONE_HOT = "one_hot"
    LABEL = "label"

class Encoder: 
    def __init__(self, column, encode_strategy: EncodingStrategy):
        self.column = column 
        self.encode_strategy = encode_strategy
        self.encoded_value = None

    def fit(self, df):
        if self.encode_strategy == EncodingStrategy.ONE_HOT:
            self.encoded_value = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.encoded_value.fit(df[[self.column]])
        elif self.encode_strategy == EncodingStrategy.LABEL:
            self.encoded_value = LabelEncoder()
            self.encoded_value.fit(df[self.column])
        return self
    
    def transform(self, df):
        df_copy = df.copy()
        if self.encode_strategy == EncodingStrategy.ONE_HOT:
            transformed = self.encoded_value.transform(df_copy[[self.column]])
            column_names = self.encoded_value.get_feature_names_out([self.column])
            encoded_df = pd.DataFrame(transformed, columns=column_names, index=df_copy.index)
            df_copy = pd.concat([df_copy, encoded_df], axis=1).drop(columns=[self.column])
        elif self.encode_strategy == EncodingStrategy.LABEL:
            df_copy[self.column] = self.encoded_value.transform(df_copy[self.column])
        return df_copy
    
class FeatureSaver:
    @staticmethod
    def save_data(train_df, test_df, output_path):

        os.makedirs(output_path, exist_ok=True)
        
        train_path = os.path.join(output_path, "train_final.csv")
        test_path = os.path.join(output_path, "test_final.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"✅ Processed data saved successfully to: {output_path}")    