from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import pinv
from ...metrics.loss_functions import mean_square_error


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) :
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of
        `self.include_intercept_`
        """
        X_with_intercept = self.add_one_column(X)
        self.coefs_ = pinv(X_with_intercept) @ y

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        X_with_intercept = self.add_one_column(X)
        return X_with_intercept @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        X_with_intercept = self.add_one_column(X)
        return mean_square_error(y, X_with_intercept @ self.coefs_)

    def add_one_column(self, X: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        Returns
        -------
        the array including the column of one;
        """
        if self.include_intercept_:
            ones_column = np.ones((X.shape[0], 1))
            X = np.column_stack((ones_column, X))
        return X

