"""Classifier Metrics"""

#Import library
import numpy as np

def accuracy_score(y_true, y_pred):
    """Calculate classification accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The true class labels.

    y_pred : array-like of shape (n_samples,)
        The predicted class labels.

    Returns
    -------
    accuracy : float
        The accuracy score.
    """
    #count the number of correct predictions
    n_true = np.sum(y_true == y_pred)
    #count the total number of samples
    n_total = len(y_true)

    #calculate accuracy
    accuracy = n_true/n_total
    
    return accuracy