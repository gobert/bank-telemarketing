# Spam classifier
Here we want to classify an email as spam or ham.

The dataset is the public corpus of apache spam assassin.

This is my work based on the book "Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems" by  Aurélien Géron.

# Requirements
* python3
* `wget`, `bunzip2`, `tar`

# Usage
* Download the dataset: `python 1-get-dataset.py`
* Start jupyter notebook: `jupyter notebook`
* Run the notebook: http://localhost:8888/notebooks/2-spam-classifier.ipynb

# Model evolution
Data prep:
- ✓ Replace numbers by NUMBER
- ✕ Delete email header
- ? Downcase
- ? Replace urls by URL
- ? Use stemming

Model evaluated: SGDClassifier, GaussianNB, SVC, KNeighborsClassifier. They all provide good results (above AUC > 90). I've sticked to SGDClassifier because it provied the best results

What could be improved:
- Being more granular on header deletion
- Evaluate all the hyperparameters of SGDClassifier and CountVectorizer
- Digger deeper into the NaiveBayes models, because they are famous for providing good results on Spam detection
- Use a harder dataset cf [1-get-dataset.py ](leads-prediction.ipynb)
