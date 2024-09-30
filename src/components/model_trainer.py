import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,X_train_transformed, X_test_transformed, y_train, y_test):
        try:
            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Random Forest Tuned":RandomForestClassifier(
                    n_estimators=50,        # Number of trees
                    max_depth=5,            # Maximum depth
                    min_samples_split=2,    # Minimum samples to split a node
                    min_samples_leaf=10,    # Minimum samples in each leaf
                    random_state=42),
                "Logistic Regression": LogisticRegression(),
                        }
    

            model_report:dict=evaluate_models(X_train=X_train_transformed,y_train=y_train,X_test=X_test_transformed,y_test=y_test,
                                             models=models)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test_transformed)

            f1socre = f1_score(y_test, predicted)
            return f1socre
            



            
        except Exception as e:
            raise CustomException(e,sys)