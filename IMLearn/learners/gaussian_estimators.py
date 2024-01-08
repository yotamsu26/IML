from __future__ import annotations

import math

import numpy as np
from numpy.linalg import inv as inv
from numpy.linalg import det as det

class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased
             estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not
             been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in
             `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in
            `UnivariateGaussian.fit`
            function.
        """

        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated
        estimation (where estimator is either biased or unbiased).
        Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = X.mean()
        if self.biased_:
            m = len(X)
        else:
            m = len(X) - 1

        self.var_ = (np.sum((X-self.mu_)**2))/m
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted
         estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")
        sqrt_part = math.sqrt(2 * np.pi * self.var_)

        exp_pat = np.exp((-1 / (2 * self.var_)) *
                         (X - self.mu_) * (X - self.mu_))
        return (1 / sqrt_part) * exp_pat

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian
         model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        vec_length = X.size
        log_part = np.log(1 / (2 * np.pi * sigma))
        sum_part = -np.sum((X - mu) ** 2) / (2 * sigma)
        return (vec_length / 2) * log_part + sum_part


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not
             been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in
             `MultivariateGaussian.fit` function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in
             `MultivariateGaussian.fit` function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to
        calculated estimation. Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = X.mean(axis=0)
        self.cov_ = np.cov(X.T)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted
         estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")
        sqrt_part = np.sqrt(((2 * np.pi) ** len(X[1])) * det(self.cov_))
        exp_part = np.exp(-0.5 * np.sum(((X-self.mu_)@inv(self.cov_)) *
                                        (X-self.mu_), axis=1))
        return (1 / sqrt_part) * exp_part

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray,
                       X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian
         model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given
             parameters of Gaussian
        """
        vec_size = len(X)  # m - according to the formula
        var_len = len(X[0])  # d - according to the formula
        new_vec = X - mu
        lop_part = np.log(((2 * np.pi) ** var_len) * det(cov))
        sum_part = np.sum(new_vec.T * (inv(cov)@new_vec.T))
        return (-vec_size / 2) * lop_part - 0.5 * sum_part
