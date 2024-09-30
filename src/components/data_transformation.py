import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            categorical_columns = [
                "awayTeamCode",
                "homeTeamCode"
            ]

            numerical_columns = [ 
                'season', 'isPlayoffGame', 'total_games_played_by_home',
                'total_games_played_by_away', 'total_wins_home', 
                'total_losses_home', 'total_wins_away', 'total_losses_away', 
                'last_10_games_win_home', 'last_10_games_win_away',
                'last_meeting_result', 'last_game_result_home', 
                'last_game_result_away'
                            ]

            num_pipeline = Pipeline (
                steps=[
                ("imputer",SimpleImputer(strategy="constant",fill_value=2))
                ]

            )

            cat_pipeline= Pipeline(
                steps=[
                ("encoder",OneHotEncoder(handle_unknown="ignore"))

                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor=ColumnTransformer(
                transformers=
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,raw_path):

        '''
        Applies the preprocessor pipeline to the training data, splitting it into training and testing datasets.
        '''

        try:
            df=pd.read_csv(raw_path)

            target_column_name = "homeTeamWon"
            X = df.drop(columns=[target_column_name], axis=1)
            y = df[target_column_name]

            # Logging
            logging.info("Splitting data into train and test sets")

            # Split the data into training and testing sets (90% train, 10% test)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

            # Obtain preprocessor object
            logging.info("Obtaining preprocessing object")
            preprocessor = self.get_data_transformer_object()

            # Transform training and testing data
            logging.info("Applying preprocessing object on training and testing data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Save the preprocessing object
            logging.info(f"Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            return X_train_transformed, X_test_transformed, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)

