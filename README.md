perceptron-python
=================

**Intuition and mathematical sketch behind algorithm:**

**Karamkars algorithms and simplex method, polynomial computation time.**

**Fontanari and Meir's genetic algorithm that figured out these rules.**

This repository contains notes on the perceptron machine learning algorithm. The python scripts are from the book Python Machine Learning by Sebastian Raschka. Meanwhile, the R script is my implementation in the program R.

Furthermore, the algorithm is then applied to the iris data set of flower species and their dependence on petal and sepal dimensions.

I have played around with the data some more in the Rmd file beyond the python scripts.

However, Raschka outlines a plotting code to visualize the descision boundary for the 2D data set. This is very useful and would be nice to implement in R.

![](https://github.com/FyzHsn/perceptron-python/blob/master/Separability_Boundary.png?raw=true)

Also, the number of misclassifications made per epoch is given by:

![](https://github.com/FyzHsn/perceptron-python/blob/master/Misclassifications.png?raw=true)

If we look at the parameters for all three species, the separability plot becomes:

![](https://github.com/FyzHsn/perceptron-python/blob/master/Separability_Boundary_II.png?raw=true)

When we try to separate between Setosa and Non-setosa, we get the following misclassification plot:

![](https://github.com/FyzHsn/perceptron-python/blob/master/Misclassifications_Setosa.png?raw=true)

Not only is it separable, it converges earlier as well. Of course more data points are used per epoch.

When, we try to separate virginica from non-virginica, we find that the errors do not go down to zero and hence, the weight does not converge:

![](https://github.com/FyzHsn/perceptron-python/blob/master/Misclassifications_Virginica.png?raw=true)

Lastly, html file of the Rmd file published on my rpubs account is [here](https://rpubs.com/FaiHas/197581).

