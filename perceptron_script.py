# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:42:11 2016

In this script, I play around with the perceptron code.
"""

import pandas as pd

"""Read in iris data from machine learning database. """
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/iris/iris.data', header=None)
    
"""View data frame head and tail. """
print(df.head())

import matplotlib.pyplot as plt
import numpy as np

"""Extract the first 100 rows of the fourth column of the data frame. """
y = df.iloc[0:100, 4].values
print(y[0:10])

y = np.where(y == "Iris-setosa", -1, 1)
X = df.iloc[0:100, [0, 2]].values

"""
Plot data before executing perceptron algorithm. We are looking for
linear separability.

"""
plt.scatter(X[0:50, 0], X[0:50, 1], 
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], 
            color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()

"""
We can see that the data is linearly separable. Now, let us run the 
perceptron algorithm and look at the number of errors make during each 
epoch.

"""
from perceptron import Perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_,
         marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()    


""" 
The following is code by Sebastian Raschka to visualize the linear
separability of the dataset.

"""








