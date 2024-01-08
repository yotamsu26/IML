import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    return np.mean((y_pred - y_true) ** 2)


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray,
                            normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    sum_of_misclassification = 0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            sum_of_misclassification += 1
    if normalize:
        return sum_of_misclassification/(len(y_true))
    return sum_of_misclassification


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    correct_samples = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct_samples += 1
    return correct_samples / len(y_true)


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    # cross_entropy_loss = 0
    # for i in range(len(y_true)):
    #     cross_entropy_loss += -1 * y_true[i] * np.log(y_pred[i])
    # return cross_entropy_loss


def softmax(X: np.ndarray) -> np.ndarray:
    """
    Compute the Softmax function for each sample in given data

    Parameters:
    -----------
    X: ndarray of shape (n_samples, n_features)

    Returns:
    --------
    output: ndarray of shape (n_samples, n_features)
        Softmax(x) for every sample x in given data X
    """
    # total_exp = 0
    # softmax_arr = np.zeros(len(X))
    # for i in range(len(X)):
    #     total_exp += np.exp(X[i])
    # for i in range(len(X)):
    #     softmax_arr[i] = np.exp(X[i])/total_exp
    # return softmax_arr

