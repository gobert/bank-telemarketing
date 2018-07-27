# Costa Rican Poverty

This is a collection of notebooks related to the kaggle competition: [costa rican household poverty prediction.](https://www.kaggle.com/c/costa-rican-household-poverty-prediction)

# Model evolution

* First try with a [**Feature selection** with Random Forest](feature-selection-with-random-forest.ipynb). The more surprising was a huge discrependcy between local F1-score and on kaggle.
* **Evaluation**: [Reproduction of kaggle F1-score](how-to-reproduces-macro-f1-score-locally.ipynb). For that we should focus on household, not individual.
* Evaluation of the [categorization of the children by age]().
* [**PCA** on house features](feature-engineering-children.ipynb) (wall, roof, ...) and check its potential predictiveness
