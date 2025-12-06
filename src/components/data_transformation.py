import pandas as pd 
import sys 
import os 
from dataclasses import dataclass
import numpy as np 

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    processor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This is responsible to perform all types of data transformation
        from Imputing, Encoding and Scaling the data
        
        '''
        try:
            numerical_features=["writing_score","reading_score"]
            categorical_features=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OneHotEncoder())                               
                ]
            )
            
            logging.info("Numerical Columns Imputing and Scaling Done")
            logging.info("Categorical Columns Encoding,Imputing and Scaling Done")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_features),
                    ("cat_pipeline",cat_pipeline,categorical_features)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read Train and Test data completed")
            logging.info("Obtaining processing Object")
            
            processor_obj=self.get_data_transformer_object()
            target_column_name="math_score"
            numerical_features=["writing_score","reading_score"]
            
            X_train=train_df.drop(columns=[target_column_name],axis=1)
            y_train=train_df[target_column_name]
            
            X_test=test_df.drop(columns=[target_column_name],axis=1)
            y_test=test_df[target_column_name]
            
            logging.info("applying preprocessing object on training dataframe and testing dataframe")
            X_train_arr=processor_obj.fit_transform(X_train)
            X_test_arr=processor_obj.transform(X_test)
            train_arr=np.c_[
                X_train_arr,np.array(y_train)
            ]
            test_arr=np.c_[
                X_test_arr,np.array(y_test)
            ]
            logging.info("Saved preprocessing object")
            
            save_object(
                file_path=self.data_transformation_config.processor_obj_file_path,
                obj=processor_obj
            )
            
            return (
                train_arr,
                test_arr, 
                self.data_transformation_config.processor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
