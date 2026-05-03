import hydra
from omegaconf import DictConfig
import pandas as pd
from data.make_dataset import CSVDataIngestion, DataSplitter

from utils.logger import get_logger

logger = get_logger(__name__)

@hydra.main(config_path="../../", config_name="config", version_base="1.2")
def data_load(cfg: DictConfig):
    try:
        logger.info(">>> Starting Data Ingestion and Splitting Stage <<<")
        
        logger.info(f"Loading data from: {cfg.paths.raw_data}")
        ingestor = CSVDataIngestion(cfg.paths.raw_data)
        df = ingestor.load_data()
        logger.info(f"Data loaded successfully with shape: {df.shape}")

        cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped unnecessary columns: {cols_to_drop}")

        logger.info(f"Splitting data with test_size={cfg.params.test_size}")
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
        logger.info("Train and Test data saved to data/interim")
        
        logger.info("Data Ingestion and Split completed successfully via Hydra!")

    except Exception as e:
        logger.error(f"Failed to complete Data Ingestion stage. Error: {str(e)}")
        raise e

if __name__ == "__main__":
    data_load()