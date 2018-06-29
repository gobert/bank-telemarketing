# Iris classification

## Visulation of linear models
In theory the 3 following models should be the same:
* SVM classification with linear kernel
* Linear classification
* SGD classification with `alpha = 1 / (len(X) * C)`

with
* `alpha` constant that multiplies the regularization term for SGD
* `C`  penalty term for the Linear Classification and SVM
* `len(X)` number of obersvation for the training


The goal here is to visualy check this assumption. We have a 2-D input features: petal length and petal width. And we will represent them on a graph besides the 3 linear models.
