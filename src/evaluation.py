import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation our models
    """
    @abstractmethod
    def calculates_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model
        :param y_true: True labels
        :param y_pred: Predicted labels
        :return:
        """
        pass

class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error
    """
    def calculates_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e

class R2(Evaluation):
    """
    Evaluation strategy that uses R2 score
    """
    def calculates_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 score")
            r2 = r2_score(y_true,y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2

        except Exception as e:
            logging.error("Error in calculating R2 score: {}".format(e))
            raise e

class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Square Error
    """
    def calculates_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info("RMSE: {}".format(rmse))
            return rmse

        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e
