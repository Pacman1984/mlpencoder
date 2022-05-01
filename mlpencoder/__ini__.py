from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from typing import Union, List, Tuple, Dict, Any
import pandas as pd
from scipy.special import expit as logistic_sigmoid

class MLPEncoder(BaseEstimator, TransformerMixin):
    # initializer 
    def __init__(self, task: str='classifiction', arch: List[int] = [60,40,20,10], activation: str='tanh') -> np.ndarray:
        self.arch = arch
        self.task = task
        self.activation = activation
        if self.task == 'classification':
            self.model = MLPClassifier(hidden_layer_sizes=self.arch, max_iter=1000, early_stopping=True)
        else:
            self.model = MLPRegressor(hidden_layer_sizes=self.arch, max_iter=1000, early_stopping=True)

    # extract deepest hidden layer representations from input
    def _deepest_layer(self, X, layer: int=0):
        L = np.matmul(X, self.model.coefs_[layer]) + self.model.intercepts_[layer]
        layer += 1
        if layer < len(self.model.hidden_layer_sizes):
            if self.activation == 'tanh':
                L = np.tanh(L)
            elif self.activation == 'sigmoid':
                L = logistic_sigmoid(L)
            elif self.activation == 'relu':
                L = np.maximum(L, 0) 
            elif self.activation == 'identity':
                L = L
            return self._deepest_layer(X=L, layer=layer)
        else:
            return L

    # fit function
    def fit(self, X, y=None):
        #FIT
        self.model.fit(X, y)

        return self

    # TRANSFORM
    def transform(self, X):
        # extract deepest hidden layer representations from input
        L = self._deepest_layer(X)

        return L