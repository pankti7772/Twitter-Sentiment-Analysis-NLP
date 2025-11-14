# src/components/model_trainer.py
import os
import sys
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import GridSearchCV

# Project utilities (fallbacks if not available)
try:
    from src.exception import CustomException
except Exception:  # pragma: no cover
    class CustomException(Exception):
        def __init__(self, original_exception, sys_info=None):
            super().__init__(str(original_exception))
            self.original_exception = original_exception
            self.sys_info = sys_info

try:
    from src.logger import logging as project_logging
    logging = project_logging
except Exception:  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    Trains multiple classifiers on text TF-IDF + meta features and selects the best model
    by macro F1 (GridSearchCV). Also logs and returns per-class metrics with emphasis on
    the Negative (toxic) class.
    """

    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config

    def _fit_and_evaluate(self, model, X_train, y_train, X_test, y_test, param_grid=None):
        """
        Fit (optionally with grid search) and evaluate a single model.
        Returns trained model and metrics dict.
        """
        try:
            if param_grid:
                gs = GridSearchCV(model, param_grid, scoring="f1_macro", cv=3, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)
                best = gs.best_estimator_
                logging.info("GridSearch best params: %s", gs.best_params_)
            else:
                best = model
                best.fit(X_train, y_train)

            y_pred = best.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            p, r, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=np.unique(y_test), zero_division=0)
            cls_report = classification_report(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            metrics = {
                "accuracy": float(acc),
                "precision_per_class": p.tolist(),
                "recall_per_class": r.tolist(),
                "f1_per_class": f1.tolist(),
                "support_per_class": support.tolist(),
                "classification_report": cls_report,
                "confusion_matrix": cm.tolist(),
                "model_object": best
            }
            return best, metrics

        except Exception as e:
            logging.exception("Error in _fit_and_evaluate")
            raise CustomException(e, sys)

    def evaluate_models(self, X_train, y_train, X_test, y_test) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Trains a set of candidate models, returns:
          - models_metrics: mapping model_name -> metrics dict
          - trained_models: mapping model_name -> trained model object
        """
        try:
            logging.info("Preparing candidate models and parameter grids")

            models = {
                "LogisticRegression": LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced", n_jobs=-1),
                "LinearSVC": LinearSVC(max_iter=5000, class_weight="balanced"),
                "MultinomialNB": MultinomialNB(),
                "RandomForest": RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)
            }

            params = {
                "LogisticRegression": {
                    "C": [0.01, 0.1, 1.0]
                },
                "LinearSVC": {
                    "C": [0.01, 0.1, 1.0]
                },
                "MultinomialNB": {
                    "alpha": [0.1, 0.5, 1.0]
                },
                "RandomForest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10]
                }
            }

            models_metrics = {}
            trained_models = {}

            for name, model in models.items():
                logging.info("Training & tuning model: %s", name)
                best_model, metrics = self._fit_and_evaluate(model, X_train, y_train, X_test, y_test, param_grid=params.get(name))
                models_metrics[name] = metrics
                trained_models[name] = best_model

                logging.info("Model %s — accuracy: %.4f", name, metrics["accuracy"])
                logging.info("Classification report for %s:\n%s", name, metrics["classification_report"])

            return models_metrics, trained_models

        except Exception as e:
            logging.exception("Error in evaluate_models")
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_arr, test_arr, feature_type: str = "array"):
        """
        :param train_arr: numpy array or sparse array with last column = label (y)
        :param test_arr: numpy array or sparse array with last column = label (y)
        :param feature_type: "array" (dense numpy) or "sparse" (scipy sparse). The code will handle both.
        :return: dict with best model name, metrics and path to saved model
        """
        try:
            logging.info("Model training orchestrator started")

            # Split features and labels; supports dense arrays or (n_samples, n_features+1)
            if hasattr(train_arr, "shape") and train_arr.shape[1] > 1:
                X_train = train_arr[:, :-1]
                y_train = train_arr[:, -1].astype(int)
            else:
                raise CustomException("train_arr must be a 2D array with last column as label", sys)

            if hasattr(test_arr, "shape") and test_arr.shape[1] > 1:
                X_test = test_arr[:, :-1]
                y_test = test_arr[:, -1].astype(int)
            else:
                raise CustomException("test_arr must be a 2D array with last column as label", sys)

            logging.info("Shapes — X_train: %s, y_train: %s, X_test: %s, y_test: %s",
                         getattr(X_train, "shape", None),
                         getattr(y_train, "shape", None),
                         getattr(X_test, "shape", None),
                         getattr(y_test, "shape", None))

            # Evaluate candidate models
            models_metrics, trained_models = self.evaluate_models(X_train, y_train, X_test, y_test)

            # Choose best model by macro F1 averaged across classes
            best_name = None
            best_f1 = -1.0
            for name, m in models_metrics.items():
                f1_scores = m["f1_per_class"]
                # compute macro f1
                macro_f1 = float(np.mean(f1_scores)) if len(f1_scores) > 0 else 0.0
                logging.info("Model %s macro F1: %.4f", name, macro_f1)
                if macro_f1 > best_f1:
                    best_f1 = macro_f1
                    best_name = name

            best_model = trained_models[best_name]
            best_metrics = models_metrics[best_name]

            logging.info("Selected best model: %s with macro F1=%.4f", best_name, best_f1)

            # Save best model (pipeline already contains any processing if needed)
            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            joblib.dump(best_model, self.config.trained_model_file_path)
            logging.info("Saved best model to %s", self.config.trained_model_file_path)

            # Display summary with special focus on Negative class (assumes label mapping known)
            # We will try to detect index of 'Negative' label if label encoder was used upstream.
            # Otherwise, we just show per-class metrics with their indices.
            summary = {
                "best_model_name": best_name,
                "best_macro_f1": best_f1,
                "best_model_path": self.config.trained_model_file_path,
                "best_model_metrics": best_metrics,
                "all_models_metrics": models_metrics
            }

            # Pretty-print key outputs
            print(f"\n✅ Best Model: {best_name} (macro F1 = {best_f1:.4f})")
            print("\n📋 Best model classification report:\n")
            print(best_metrics["classification_report"])
            print("\n📊 Confusion Matrix:\n")
            print(np.array(best_metrics["confusion_matrix"]))

            return summary

        except Exception as e:
            logging.exception("Error in initiate_model_trainer")
            raise CustomException(e, sys)
