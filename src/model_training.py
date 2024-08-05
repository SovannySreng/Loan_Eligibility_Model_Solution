
# src/model_training.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_logistic_regression(xtrain, ytrain):
    lrmodel = LogisticRegression().fit(xtrain, ytrain)
    return lrmodel

def train_random_forest(xtrain, ytrain):
    rfmodel = RandomForestClassifier(n_estimators=100, 
                                     min_samples_leaf=5, 
                                     max_features='sqrt')  # Change 'auto' to 'sqrt'
    rfmodel.fit(xtrain, ytrain)
    return rfmodel