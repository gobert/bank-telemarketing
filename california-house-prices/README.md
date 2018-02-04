Here we want to predict the prices of the california houses according to different other features.

The dataset is the "California Housing Prices" from StatLib.

This is my work based on the book "Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems" by  Aurélien Géron.

# Precision of the model: evolution
The model evolved in 3 steps:
* First a dummy linear model
* Then choosing  a more complicated models. RandomForest was the best
* The fine tuning the model

This shows how the precision (RMSE) evolved:

![precision of the model: evolution](https://user-images.githubusercontent.com/1684807/35779533-c3eeedf2-09ce-11e8-929e-3ced1280d108.png)
