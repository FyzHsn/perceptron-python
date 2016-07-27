perceptron-python
=================

## **Intuition and mathematical sketch behind algorithm:**

### Definitions:  

1.  Define input and weight vectors. Following that define extended input and weight vectors.  
2.  Define open (closed) positive and negative half-spaces such that the net input, i.e. dot product of the weight and input vectors are > (>=) or < (<=) 0, respectively.  
3.  Define linear separability of the input data points with binary classification as belonging to sets A and B respectively. Also, at this point it can be proven that linear separability leads to absolute linear separability for finite sets. 


### Describe the algorithm:

*start:* The initial weight vector is randomly generated at t:=0.

*test:* A vector from the union of the negative and the positive half space is chosen randomly.  
        if classified correctly, go back to *test*,  
        if classified incorrectly, go to *update*.  
        
*update:*  Add/subtract the misclassified positive/negative point to the weight vector and update t:=t+1, go to *test*.  

### Sketch of convergence proof:

**Karamkars algorithms and simplex method leads to polynomial computation time.**

**Fontanari and Meir's genetic algorithm also figured out these rules.**

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

