import sys 
import os 
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

class TrainPipeline:
    def __init__(self):
        self.data_ingestion=DataIngestion()
        self.data_transformation=DataTransformation()
        self.model_trainer=ModelTrainer()
        
    def run(self):
        try:
            logging.info("Model Training started")
            train_data_path,test_data_path=self.data_ingestion.initiate_data_ingestion()
            train_arr,test_arr,preprocessor_path=self.data_transformation.initiate_data_transformation(train_data_path,test_data_path)
            r2=self.model_trainer.initiate_model_trainer(train_arr,test_arr)
            return r2
        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    pipeline=TrainPipeline()
    pipeline.run()
    