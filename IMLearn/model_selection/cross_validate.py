from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated
        model. When called, the scoring function receives the true- and
        predicted values for each sample and potentially additional arguments.
        The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    partition_size = int(len(X)/cv)
    ts, vs = 0, 0
    for k in range(cv):
        first_part = k*partition_size
        second_part = k*partition_size + partition_size
        ts_X = np.concatenate((X[:first_part], X[second_part:]), axis=0)
        vs_X = X[first_part:second_part]
        ts_Y = np.concatenate((y[:first_part], y[second_part:]), axis=0)
        vs_Y = y[first_part:second_part]

        estimator.fit(ts_X, ts_Y)

        ts += scoring(estimator.predict(ts_X), ts_Y)
        vs += scoring(estimator.predict(vs_X), vs_Y)

    return ts/cv, vs/cv
