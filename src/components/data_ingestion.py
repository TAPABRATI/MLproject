import os
import sys
#import sys
#sys.path.append('src/package1')
from src.exception import CustomException
from src.loggers import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifact','train.csv')
    test_data_path = os.path.join('artifact','test.csv')
    raw_data_path = os.path.join('artifact','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def iniatiate_data_ingestion(self):
        logging.info("Data Ingestion Started!!")
        try:
            df = pd.read_csv('F:/class/ML/notebook/data/stud.csv')
            logging.info("read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=44)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)
            logging.info("train test split is done and saved the dfs")
            logging.info("Data ingestion completed!!")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data, = obj.iniatiate_data_ingestion()
    print(train_data,test_data)
    data_transform = DataTransformation()
    train_arr,test_arr,_ = data_transform.initiate_data_transformation(train_data,test_data)
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))