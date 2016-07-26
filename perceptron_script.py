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
plt.savefig('Misclassifications.png') 
plt.clf()  


""" 
The following is code by Sebastian Raschka to visualize the linear
separability of the dataset.

"""
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)                      
                    
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.savefig('Separability_Boundary.png')                
plt.clf()    

"""
I carry out some more exploratory data analysis, by keeping all the 
species of the iris data set.

"""
y = df.iloc[:, 4].values
X = df.iloc[:, [0, 2]].values

"""
Rows 0 - 50 (Iris-setosa), 50 - 100 (Iris-versicolor), 100 - 150
(Iris-virginica)

"""

# Scatter plot
plt.scatter(X[0:50, 0], X[0:50, 1], 
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')
plt.scatter(X[100:151, 0], X[100:151, 1],
            color='cyan', marker='^', label='versicolor')       
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.savefig('Separability_Boundary_II.png')
plt.clf()

# Run the perceptron data to differentiate between setosa and others.            
y = np.where(y == 'Iris-setosa', -1, 1)

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_,
         marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.title('Setosa vs Non-setosa')
plt.savefig('Misclassifications_Setosa.png')
plt.clf()

""" 
Meanwhile we cannot distinguish virginica from the others due to the
lack of linear separability.

"""
y = df.iloc[:, 4].values
y = np.where(y == 'Iris-virginica', -1, 1)

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_,
         marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.title('Virginica vs Non-virginica')
plt.savefig('Misclassifications_Virginica.png')
plt.clf()


