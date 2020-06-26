# machine_learning.py

from typing import List, Set, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import surprise
from surprise import Dataset
from surprise import AlgoBase, SVD, KNNBaseline
from surprise.reader import Reader
from surprise.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

CLASSIFICATION_SCORE_FUNCS_DICT = {
    "accuracy_score": accuracy_score,
    "roc_auc_score": roc_auc_score,
    "f1_score": f1_score,
    "precision_score": precision_score,
    "recall_score": recall_score,
    "confusion_matrix": confusion_matrix
}


class LovePredictor:
    rating_scale = (0, 1)

    def __init__(self, love_matrix: np.array, model: AlgoBase,
                 test_size: float = 0.2):
        self.love_matrix = love_matrix
        self.model = model
        self.reader = Reader(rating_scale=self.rating_scale)
        self.test_size = test_size

    # Data handeling
    def pandas_data(self, love_matrix: np.array, inplace: Optional[bool] = True):
        df_love = pd.DataFrame(love_matrix)
        df_love_long = wide_to_long(df_love, drop_na=False)

        if inplace:
            self.df_love = df_love
            self.df_love_long = df_love_long
        else:
            return df_love, df_love_long

    def surprise_data(self, df_long: Optional[pd.DataFrame], reader: Optional[Reader],
                      inplace: Optional[bool] = True):

        if not df_long:
            df_long = self.df_love
        if not reader:
            reader = self.reader

        surprise_dataset = Dataset.load_from_df(
            df_long, reader
        )

        if inplace:
            self.surprise_dataset = surprise_dataset
        else:
            return surprise_dataset

    @staticmethod
    def _love_long_split(df_love_long):
        filled = df_love_long["Value"].notnull()
        df_love_long_filled = df_love_long.loc[filled, :]
        df_love_long_nan = df_love_long.loc[~filled, :]

        return df_love_long_filled, df_love_long_nan

    # Splitting data
    def split_train_test(self, surprise_dataset: Optional[Dataset], test_size: Optional[float], inplace: Optional[bool]):
        if not surprise_dataset:
            surprise_dataset = self.surprise_dataset
        if not test_size:
            test_size = self.test_size

        trainset, testset = train_test_split(surprise_dataset, test_size)

        if inplace:
            self.trainset = trainset
            self.testset = testset
        else:
            return trainset, testset

    # Model fit
    def fit(self, model: Optional[AlgoBase], trainset: Optional[surprise.trainset.Trainset], inplace: Optional[bool]):
        if not model:
            model = self.model
        if not trainset:
            trainset = self.trainset

        fit_model = model.fit(trainset)

        if inplace:
            self.model = fit_model
        else:
            return fit_model

    # Model predict
    @staticmethod
    def df_long_to_surprise_predictset_iterator(df_long):
        predictset_iterator = zip(
            df_long["Row"].values,
            df_long["Column"].values,
            df_long["Value"].values
        )

        return predictset_iterator

    def get_predictions(self, model: Optional[AlgoBase], predictset: List[Tuple]):
        try:
            prediction_list = model.test(predictset)
        except AttributeError as e:
            print(f"Model fit failed for {model}.")
            print("Did you already fit the model?")
            raise e

        return prediction_list

    @staticmethod
    def surprise_predictions_to_df(predictions_surp):
        values = [(x.uid, x.iid, x.r_ui, x.est) for x in predictions_surp]

        df_predictions = pd.DataFrame(
            data=values,
            columns=["Row", "Column", "y", "probabilities"]
        )

        return df_predictions

    @classmethod
    def round_probas(cls, df_predictions):
        low = cls.rating_scale[0]
        high = cls.rating_scale[1]

        rating_middle = (high - low)/2
        df_predictions["y_hat"] = df_predictions["probabilities"].copy()

        round_up_bool = df_predictions.loc[:, "probabilities"] > rating_middle
        round_down_bool = df_predictions.loc[:, "probabilities"] <= rating_middle

        df_predictions.loc[round_up_bool, "y_hat"] = high
        df_predictions.loc[round_down_bool, "y_hat"] = low

        return df_predictions


# Model evaluation
def compute_classification_scores(y, y_hat, classification_score_funcs_dict=CLASSIFICATION_SCORE_FUNCS_DICT):
    classification_scores = {
        k: v(y, y_hat)
        for k, v in classification_score_funcs_dict.items()
    }

    return classification_scores


# Merge original data with predictions
# TODO: Make option for setting self.<...>
def merge_original_with_missing_predictions(self, df_love_long, df_missing_long):
    love_matrix_filled_long_np = np.concatenate([
        df_love_long.loc[:, ["Row", "Column", "Value"]].values,
        df_missing_long.loc[:, ["Row", "Column", "y_hat"]].values
    ])

    df_love_matrix_filled_long = pd.DataFrame(
        love_matrix_filled_long_np, columns=["Row", "Column", "Value"]
    )

    return df_love_matrix_filled_long


def wide_to_long(df_wide, column_names=['Row', 'Column', 'Value'], drop_na=True):
    df_wide_transp = df_wide.transpose()
    df_long = pd.DataFrame(df_wide_transp.copy().unstack()).reset_index()
    df_long.columns = column_names

    if drop_na:
        df_long = df_long.dropna()

    return df_long
