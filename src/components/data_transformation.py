import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.loggers import logging
from src.utils import save_object
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifact','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        this will be responsible for data transformation
        """
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[("imputer",SimpleImputer(strategy="median")),
                       ("scalar", StandardScaler())]

            )

            cat_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scalar", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical Columns:{categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocessor = ColumnTransformer([("num_pipeline", num_pipeline,numerical_columns),
                                             ("cat_pipeline",cat_pipeline,categorical_columns)])
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("we are inside intiate data transformation !!")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("read train test data completed")

            preprocessing_obj = self.get_data_transformation_object()
            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]  
            logging.info("done the seperation of indep and dep variable")
            logging.info("applying preprocessing obj on train test data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file,
                obj = preprocessing_obj
            )

            return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file

            )
        except Exception as e:
            raise CustomException(e,sys)


        
