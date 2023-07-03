"""Classifier Adaboost"""

#Import library
import copy
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class DecisionStump:
    """Decision stump classifier based class.

    Parameters
    ----------
    polarity : {1, -1}, default = 1
        The polarity of the stump prediction.
        If the feature value at `feature_idx` is greater than the `threshold`,
        it predicts the `polarity` value. Otherwise, it predicts the opposite of `polarity`.
    
    feature_idx : int or None, default = None
        The index of the feature used to split a node.
        If None, the feature index will be determined during the fitting process.

    threshold : float or None, default = None
        The threshold value used to split a node on the specific feature.
        If None, the threshold will be determined during the fitting process.

    Attributes
    ----------
    polarity : {1, -1}
        The polarity of the stunp prediction.

    feature_idx : int or None
        The index of the feature used to split a node.

    threshold : float or None
        The threshold value used to split a node on the specific feature.

    
    Methods
    -------
    fit(X, y, weights)
        Fit the decision stump using the provided training data.

    predict(X)
        Predict the class labels for the input samples.
    """
    def __init__(
        self,
        polarity=1,
        feature_idx=None,
        threshold=None
    ):
        #Initialize attributes with default values or values given at object creation
        self.polarity = polarity
        self.feature_idx = feature_idx
        self.threshold = threshold

    def _best_split(self, X, y, weights):
        """Find the best split for a node
        
        Fundamental assumtion
            - y_pred where X[:, feature_i] > threshold = polarity
            - Otherwise opposite of polarity
                - if polarity = 1, the opposite = -1
                - if polarity = -1, the opposite = 1
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.
        
        weights : array-like of shape (n_samples,)
            The sample weights.

        Returns
        -------
        best_polarity : {1, -1}
            The polarity of the best split.

        best_feature_idx : int
            The index of the feature that provides the best split.

        best_threshold : float
            The threshold value that provides the best split.
        """

        #Intialize
        best_polarity = None
        best_feature_idx = None
        best_threshold = None
        min_error = float('inf')

        for feature_i in range(self.n_features):
            #Extract data
            X_column = X[:, feature_i]

            #Determine the unique threshold value of the feature column
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                #Generate the predictions
                #Assume y_pred where X[:, feature_i] <= threshold is -1
                polarity = 1
                predictions = np.ones(self.n_samples)
                predictions[X_column <= threshold] = -1

                #Calculate the amount of error based on the difference between the prediction and the actual label
                error = np.sum(weights[y != predictions])

                #If the error is greater than 0.5, invert polarity and error
                if error > 0.5:
                    error = 1 - error
                    polarity = -1
                
                #Update the best splitting value if a lower error is found
                if error < min_error:
                    best_polarity = polarity
                    best_feature_idx = feature_i
                    best_threshold = threshold
                    min_error = error

        return best_polarity, best_feature_idx, best_threshold

    def fit(self, X, y, weights):
        """Fit the decision stump using the provided training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        y : array-like of shape (n_samples,)
            The target values.

        weights : array-like of shape (n_samples,)
            The sample weights.
        """

        #Preparation
        X = np.array(X).copy()
        y = np.array(y).copy()
        self.n_samples, self.n_features = X.shape

        #Find the best split for the node
        best_split_result = self._best_split(X, y, weights)

        #Update attributes based on the best split result
        self.polarity = best_split_result[0]
        self.feature_idx = best_split_result[1]
        self.threshold = best_split_result[2]
    
    def predict(self, X):
        """Predict the class labels for the input samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predict class labels.
        """

        #Convert input data
        X = np.array(X).copy()
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]

        #Generate prediction
        y_pred = np.ones(n_samples) 
        if self.polarity == 1:
            y_pred[X_column <= self.threshold] = -1
        else:
            y_pred[X_column >= self.threshold] = -1
        
        return y_pred

class DecisionTreeMaxDepth1:
    """Decision tree classifier with maximum depth of 1.

    Attributes
    ----------
    tree : DecisionTreeClassifier
        The underlying decision tree classifier.

    Methods
    -------
    fit(X, y, weights)
        Fit the decision tree using the provided training data.

    predict(X)
        Predict the class labels for the input samo
    """
    def __init__(
        self
    ):
        """Intialize the DecisionTreeMaxDepth1 object.
        
        This creates an instance of the DecisionTreeClassifier with a maximum depth of 1.
        """
        self.tree = DecisionTreeClassifier(max_depth=1)

    def fit(self, X, y, weights):
        """Fit the decision tree using the provided training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        y : array-like of shape (n_samples,)
            The target values.
            
        weights : array-like of shape (n_samples,)
            The sample weights.
        """
        self.tree.fit(X, y, sample_weight=weights)

    def predict(self, X):
        """Predict the class labels for the input samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns    
        y_pred : array-like of shape (n_samples,)
            The predict class labels.
        """
        return self.tree.predict(X)

class AdaboostClassifier:
    """Adaboost classifier.
    
    Parameters
    ----------
    estimator : object or None, default = None
        The base estimator to be used in the ensemble.
        If None, a DecisionStump will be used as the default base estimator.
    
    n_estimators : int, default = 5
        The number of estimators in the ensemble.
    
    learning_rate : float, default = 1.0
        The learning rate for the Adaboost algorithm.

    Methods
    -------
    fit(X, y)
        Fit the Adaboost classifier using the provided training data.
    
    predict(X)
        Predict the class labels for the input samples.
    """
    def __init__(
        self,
        estimator=None,
        n_estimators=5,
        learning_rate=1.0,        
    ): 
        """Initialize the Adaboostclassifier object.
        
        Parameters
        ----------
        estimator : object or None, default = None
            The base estimator to be used in the ensemble.
            If None, a DecisionStump will be used as the default base estimator.
        
        n_estimators : int, default = 5
            The number of estimators in the ensemble.
        
        learning_rate : float, default = 1.0
            The learning rate for the Adaboost algorithm.
        """
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate


    def fit(self, X, y):
        """Fit the Adaboost classifier using the provided training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        y : array-like of shape (n_samples,)
            The target values.
        """
        #Check the base estimator
        #If no estimator is given, use DecisionStump as the base estimator
        if self.estimator is None:
            base_estimator = DecisionStump()
        else:
            base_estimator = DecisionTreeMaxDepth1()

        #Convert input and output data
        X = np.array(X).copy()
        y = np.array(y).copy()
        self.n_samples, self.n_features = X.shape

        #Initialize
        self.weights = np.full(self.n_samples, (1/self.n_samples))
        self.alpha = np.zeros(self.n_estimators)
        self.estimators = []    

        #Start training
        for i in range(self.n_estimators):
            #Create the estimator
            estimator = copy.deepcopy(base_estimator)

            #Train the estimator using the updated sample weights
            estimator.fit(X, y, weights = self.weights)

            #Predict & calculate the weighted error
            y_pred = estimator.predict(X)
            error = np.sum(self.weights[y != y_pred])

            #Update the alpha
            eps = 1e-10
            alpha = self.learning_rate * np.log((1-error) / (error+eps))

            #Update weights and normalization
            self.weights *= np.exp(-alpha * y * y_pred)
            self.weights /= np.sum(self.weights)

            #Append the model and alpha
            self.estimators.append(estimator)
            self.alpha[i] = alpha

    def predict(self, X):
        """Predict the class labels for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predict class labels.
        """
        #Convert the input
        X = np.array(X).copy()

        #Intialize
        y_pred = np.zeros(X.shape[0])

        #Combine the predictions of each estimator with its weights
        pred = [self.alpha[i] * self.estimators[i].predict(X) for i in range(self.n_estimators)]
        y_pred = np.sum(pred, axis=0)
        #Do the sign function
        y_pred = np.sign(y_pred)
        return y_pred