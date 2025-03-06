import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numeric_columns):
        """
        This function creates a preprocessing pipeline for all numeric features.
        """
        try:
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numeric_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data read successfully.")

            # Drop the 'id' column if it exists
            if "id" in train_df.columns:
                train_df.drop(columns=["id"], inplace=True)
            if "id" in test_df.columns:
                test_df.drop(columns=["id"], inplace=True)

            # Convert target column 'diagnosis' to numeric values: M -> 1, B -> 0
            train_df["diagnosis"] = train_df["diagnosis"].map({"M": 1, "B": 0})
            test_df["diagnosis"] = test_df["diagnosis"].map({"M": 1, "B": 0})

            target_column_name = "diagnosis"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Since all features are numeric, list them all
            numeric_columns = input_feature_train_df.columns.tolist()
            logging.info(f"Numeric columns for transformation: {numeric_columns}")

            # Create preprocessing object
            logging.info("Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object(numeric_columns)

            # Apply the preprocessing pipeline on training and test features
            logging.info("Applying preprocessing object on training and testing dataframes.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the transformed features with the target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object for later use in prediction
            logging.info("Saving the preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
