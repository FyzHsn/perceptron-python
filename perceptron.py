# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 18:46:52 2016

Implementation of Frank Rosenblatt's perceptron rule and application to the 
Iris dataset from Machine Learning with Python by Sebastian Raschka.
"""

import numpy as np
class Perceptron(object):
    """Perceptron classifier.
    
    Parameters
    ----------
    eta : float
        Learning rate between 0.0 and 1.0.
    n_iter : int
        Number of passes over the training dataset.
    
    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting. Underscore after a variable name indicates 
        that the variable was not created on instantiation of the object.
    errors_ : list
        Number of incorrect classifications every epoch.
        
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """Fit training data according to the perceptron algorithm.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training dataset, where n_samples is the number of samples
            and n_features is the number of features.
        y : {array-like}, shape = [n_samples]
            Binary classification of dataset.
                    
        Returns
        -------
        self : object
        
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[1] += update
                errors += int(update != 0)
            self.errors_.append(errors)
        return self
        
    def net_input(self, X):
        """Calculate the dot product of the features and the weights. """
        return np.dot(X, self.w_[1:]) + self.w_[0]
        
    def predict(self, X):
        """Return class label by using the Heaviside activation
        function. """        
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        
        
        
        
        
        
        
        
        
    
