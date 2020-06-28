# machine_learning.py

import sys
from pathlib import Path
# import logging  # TODO implement
from typing import List, Set, Dict, Tuple, Optional, Iterator
import numpy as np
import pandas as pd
import surprise
from surprise import Dataset
from surprise import (
    AlgoBase, NormalPredictor, BaselineOnly, KNNBasic,
    KNNWithMeans, KNNWithZScore, KNNBaseline, SVD, SVDpp,
    NMF, SlopeOne, CoClustering
)
from surprise.reader import Reader
from surprise.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

PROJECT_DIR = str(Path(__file__).resolve().parents[1])

sys.path.insert(1, PROJECT_DIR+"src")
import tindar

MIN_ROC_AUC_SCORE = 0.75

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
    def pandas_data(self, love_matrix: Optional[np.array] = None, inplace: Optional[bool] = True):  # -> Optional[Tuple]
        if love_matrix is None:
            love_matrix = self.love_matrix

        df_love = pd.DataFrame(love_matrix)
        df_love_long = wide_to_long(df_love, drop_na=False)

        if inplace:
            self.df_love = df_love
            self.df_love_long = df_love_long
        else:
            return df_love, df_love_long

    def surprise_data(self, df_long: Optional[pd.DataFrame] = None, reader: Optional[Reader] = None,
                      inplace: Optional[bool] = True):  # -> Optional[Dataset]

        if df_long is None:
            df_long = self.df_love_long
        if reader is None:
            reader = self.reader

        surprise_dataset = Dataset.load_from_df(
            df_long, reader
        )

        if inplace:
            self.surprise_dataset = surprise_dataset
        else:
            return surprise_dataset

    @staticmethod
    def love_long_split(df_love_long: pd.DataFrame):  # -> Tuple(pd.DataFrame, pd.DataFrame)
        filled = df_love_long["Value"].notnull()
        df_love_long_filled = df_love_long.loc[filled, :]
        df_love_long_nan = df_love_long.loc[~filled, :]

        return df_love_long_filled, df_love_long_nan

    # Splitting data
    def split_train_test(self, surprise_dataset: Optional[Dataset] = None,
                        test_size: Optional[float] = None, inplace: Optional[bool] = True):  # -> Optional[Tuple]
        if surprise_dataset is None:
            surprise_dataset = self.surprise_dataset
        if test_size is None:
            test_size = self.test_size

        trainset, testset = train_test_split(surprise_dataset, test_size)

        if inplace:
            self.trainset = trainset
            self.testset = testset
        else:
            return trainset, testset

    # Model fit
    def fit(self, model: Optional[AlgoBase] = None, trainset: Optional[surprise.trainset.Trainset] = None,
            inplace: Optional[bool] = True):  # -> Optional[AlgoBase]
        if model is None:
            model = self.model
        if trainset is None:
            trainset = self.trainset

        fit_model = model.fit(trainset)

        if inplace:
            self.model = fit_model
        else:
            return fit_model

    # Model predict
    @staticmethod
    def df_long_to_surprise_predictset_iterator(df_long: pd.DataFrame):  # -> pd.DataFrame
        predictset_iterator = zip(
            df_long["Row"].values,
            df_long["Column"].values,
            df_long["Value"].values
        )

        return predictset_iterator

    def predict(self, predictset: List[Tuple], model: Optional[AlgoBase] = None):  # -> List
        if model is None:
            model = self.model

        try:
            prediction_list = model.test(predictset)
        except AttributeError as e:
            print(f"Model fit failed for {model}.")
            print("Did you already fit the model?")
            raise e

        return prediction_list

    @staticmethod
    def surprise_predictions_to_df(predictions_surp: List):  # -> pd.DataFrame
        values = [(x.uid, x.iid, x.r_ui, x.est) for x in predictions_surp]

        df_predictions = pd.DataFrame(
            data=values,
            columns=["Row", "Column", "y", "probabilities"]
        )

        return df_predictions

    @classmethod
    def round_probas(cls, probs_series: pd.Series):  # -> pd.Series
        low = cls.rating_scale[0]
        high = cls.rating_scale[1]

        rating_middle = (high - low)/2

        y_hat = probs_series.copy()
        y_hat.name = "y_hat"
        y_hat[y_hat > rating_middle] = high
        y_hat[y_hat <= rating_middle] = low

        return y_hat

    # TODO: implement to follow sklearn
    def score():
        pass


# Model evaluation
def compute_classification_scores(y, y_hat, classification_score_funcs_dict=CLASSIFICATION_SCORE_FUNCS_DICT):  # -> Dict
    classification_scores = {
        k: v(y, y_hat)
        for k, v in classification_score_funcs_dict.items()
    }

    return classification_scores


# Merge original data with predictions
# TODO: Make option for setting self.<...>
def merge_filled_with_missing_predictions(df_love_long_filled: pd.DataFrame, df_missing_long: pd.DataFrame):  # -> pd.DataFrame
    love_matrix_filled_long_np = np.concatenate([
        df_love_long_filled.loc[:, ["Row", "Column", "Value"]].values,
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


def fill_love_matrix(love_matrix, model, test_size):
   # Initialize love predictor object
    lovepredictor = LovePredictor(love_matrix, model, test_size)

    # Load data
    print("------------------------------------")
    print("Loading data...")
    lovepredictor.pandas_data(inplace=True)
    lovepredictor.surprise_data(inplace=True)

    # Split filled part of love_matrix from nan part
    df_love_long_filled, df_love_long_nan = lovepredictor.love_long_split(lovepredictor.df_love_long)
    surprise_dataset_filled = lovepredictor.surprise_data(df_love_long_filled, inplace=False)

    # Split the nonnull dataframe into train and test set
    print("------------------------------------")
    print("Splitting data into train and test to evaluate model performance...")
    trainset, testset = lovepredictor.split_train_test(surprise_dataset_filled, inplace=False)

    # Fit model
    print("------------------------------------")
    print("Fitting model...")
    lovepredictor.fit(trainset=trainset)

    # Predict
    # Probabilities
    print("------------------------------------")
    print("Predicting...")
    trainset_iterable = trainset.build_testset()

    prediction_list_train = lovepredictor.predict(predictset=trainset_iterable)
    prediction_list_test = lovepredictor.predict(predictset=testset)

    df_train_probas = lovepredictor.surprise_predictions_to_df(prediction_list_train)
    df_test_probas = lovepredictor.surprise_predictions_to_df(prediction_list_test)

    # Check
    assert len(df_test_probas) + len(df_train_probas) == len(df_love_long_filled), \
       f"len(df_test_probas) + len(df_train_probas) = {len(df_test_probas) + len(df_train_probas)} "\
       f"len(df_love_long) {len(df_love_long_filled)} "\

    # Labels (0, 1)
    y_hat_train = lovepredictor.round_probas(df_train_probas["probabilities"])
    y_hat_test = lovepredictor.round_probas(df_test_probas["probabilities"])

    train_preds_np = np.concatenate([df_train_probas.values, y_hat_train.values.reshape((-1, 1))], axis=1)
    test_preds_np = np.concatenate([df_test_probas.values, y_hat_test.values.reshape((-1, 1))], axis=1)

    df_train_preds = pd.DataFrame(
        train_preds_np, index=df_train_probas.index, columns=list(df_train_probas.columns)+["y_hat"]
    )
    df_test_preds = pd.DataFrame(
        test_preds_np, index=df_test_probas.index, columns=list(df_test_probas.columns)+["y_hat"]
    )

    # Evaluate
    print("------------------------------------")
    print("Evaluating...")
    classification_scores_train = compute_classification_scores(
        df_train_preds["y"], df_train_preds["y_hat"], CLASSIFICATION_SCORE_FUNCS_DICT
    )
    classification_scores_test = compute_classification_scores(
        df_test_preds["y"], df_test_preds["y_hat"], CLASSIFICATION_SCORE_FUNCS_DICT
    )

    print("Model evaluation results:")
    print("----------------------------")
    print("TRAIN")
    for k, v in classification_scores_train.items():
        print(f"{k}: {v}")
    print("----------------------------")
    print("TEST")
    for k, v in classification_scores_test.items():
        print(f"{k}: {v}")

    if classification_scores_test["roc_auc_score"] < MIN_ROC_AUC_SCORE:
        raise Exception(f"ROC_AUC score for test set too low (should be >{MIN_ROC_AUC_SCORE}")
    else:
        print(f"Model performance test passed!")

    # Accept model and fit again
    print("------------------------------------")
    print("Repeating process on full dataset...")
    surprise_full_trainset = lovepredictor.surprise_dataset.build_full_trainset()
    lovepredictor.fit(trainset=surprise_full_trainset)

    # Predict the missing indices
    missing_indices = zip(
        df_love_long_nan["Row"].values,
        df_love_long_nan["Column"].values,
        df_love_long_nan["Value"].values
    )

    missing_predictions_list = lovepredictor.predict(missing_indices)
    df_missing_probas = lovepredictor.surprise_predictions_to_df(missing_predictions_list)
    y_hat_missing = lovepredictor.round_probas(df_missing_probas["probabilities"])

    print(f"df_missing_probas.shape: {df_missing_probas.shape}")
    print(f"len(y_hat_missing): {len(y_hat_missing)}")

    missing_preds_np = np.concatenate([df_missing_probas, y_hat_missing.values.reshape((-1, 1))], axis=1)
    df_missing_preds = pd.DataFrame(
        missing_preds_np, index=df_missing_probas.index, columns=list(df_missing_probas.columns)+["y_hat"]
    )
    print(f"df_missing_preds.shape: {df_missing_preds.shape}")


    # Merge original with prediction of missing values for final result
    print("------------------------------------")
    print("Merging original data with predictions on missing parts...")

    df_love_long_result = merge_filled_with_missing_predictions(
        df_love_long_filled, df_missing_preds
    )
    print(f"df_love_long_result.shape: {df_love_long_result.shape}")
    assert df_love_long_result.shape == (n**2, 3)

    df_love_matrix_filled = df_love_long_result.pivot(index="Row", columns="Column", values="Value")
    print(f"df_love_matrix_filled.shape: {df_love_matrix_filled.shape}")
    assert df_love_matrix_filled.shape == (n, n)

    love_matrix_filled = df_love_matrix_filled.values

    print("------------------------------------")
    print("Filling missing love data completed!")
    print(love_matrix_filled)

    # checks
    assert df_love_matrix_filled.isnull().sum().sum() == 0
    assert ((lovepredictor.df_love == df_love_matrix_filled) | lovepredictor.df_love.isnull()).all().all()
    assert love_matrix_filled.shape == love_matrix.shape

    return love_matrix_filled


MODEL_NAME_MAPPING = {
    "NormalPredictor": NormalPredictor,
    "BaselineOnly": BaselineOnly,
    "KNNBasic": KNNBasic,
    "KNNWithMeans": KNNWithMeans,
    "KNNWithZScore": KNNWithZScore,
    "KNNBaseline": KNNBaseline,
    "SVD": SVD,
    "SVDpp": SVDpp,
    "NMF": NMF,
    "SlopeOne": SlopeOne,
    "CoClustering": CoClustering
}


def make_model(model_name, model_hyperparameters):
    try:
        model = MODEL_NAME_MAPPING[model_name](**model_hyperparameters)
    except KeyError:
        raise KeyError(f"model_name {model_name} is not allowed. "
                       f"choose from {list(MODEL_NAME_MAPPING.keys())}")
    except TypeError:
        raise TypeError(f"model_hyperparameters {model_hyperparameters} not allowed "
                        f"for model {model_name}, check surprise documentation for model __init__")

    return model


if __name__ == "__main__":

    # Generate tindar problem
    n = 500
    tindar_problem = tindar.TindarGenerator(
        n, nan_probability=0.3, generation_kind="interesting",
        attractiveness_distr="uniform", unif_low=0.3, unif_high=0.8
    )
    tindar_problem.create_love_matrix()
    love_matrix = tindar_problem.love_matrix

    print(love_matrix)

    # Initialize a model
    model_name = "SVD"
    model_hyperparameters = {"n_factors": 1}
    model = make_model(model_name, model_hyperparameters)

    # Set test size to check model performance
    test_size = 0.2

    # Main script
    try:
        love_matrix_filled = fill_love_matrix(love_matrix, model, test_size)
    except Exception as e:
        print(e)
