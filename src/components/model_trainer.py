import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

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
                "Random Forest Tuned": RandomForestClassifier(
                    max_depth= 10,
                    n_estimators= 100),
                #"Random Forest Tuned2":RandomForestClassifier(
                #    n_estimators=50,        # Number of trees
                #    max_depth=5,            # Maximum depth
                #    min_samples_split=2,    # Minimum samples to split a node
                #    min_samples_leaf=10,    # Minimum samples in each leaf
                #   random_state=42),
                #"Logistic Regression": LogisticRegression(
                #    C = 10,
                #    class_weight= 'balanced', 
                #    solver = 'liblinear'
                #),
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

            # Calculate metrics
            f1 = f1_score(y_test, predicted, average='weighted')
            accuracy = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted)
            recall = recall_score(y_test, predicted)
            cm = confusion_matrix(y_test, predicted)

            # Log the metrics
            logging.info(f"F1 Score: {f1:.4f}")
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"Confusion Matrix:\n{cm}")

            return f1
            
        except Exception as e:
            raise CustomException(e,sys)