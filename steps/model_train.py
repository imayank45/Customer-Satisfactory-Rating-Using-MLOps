import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        config: ModelNameConfig
) -> RegressorMixin:
    """
    Trains the model on the ingested data.

    :param X_train: pd.DataFrame,
    :param X_test: pd.DataFrame,
    :param y_train: pd.DataFrame,
    :param y_test: pd.DataFrame
    """
    model = None

    try:
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model

        else:
            raise ValueError("Model {} not supported".format(config.model_name))

    except Exception as e:
        logging.error("Error in training the model: {}".format(e))
        raise e
