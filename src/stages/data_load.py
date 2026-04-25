import hydra
from omegaconf import DictConfig
import pandas as pd
from data.make_dataset import CSVDataIngestion, DataSplitter

@hydra.main(config_path="../../", config_name="config", version_base="1.2")
def data_load(cfg: DictConfig):
    
    ingestor = CSVDataIngestion(cfg.paths.raw_data)
    df = ingestor.load_data()

    cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=cols_to_drop)

    splitter = DataSplitter()
    X_train, X_test, y_train, y_test = splitter.split_data(
        df, 
        target_col=cfg.params.target, 
        test_size=cfg.params.test_size, 
        random_state=cfg.params.random_state
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    splitter.save_data(train_df, test_df, "data/interim")
    
    print(" Data Ingestion and Split completed via Hydra!")

if __name__ == "__main__":
    data_load()