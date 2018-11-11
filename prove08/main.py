from sklearn import datasets
from load_b_cancer import load_b_cancer
from load_fire import load_fire
from load_mpg import load_mpg
from load_student import load_student

# http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

# LOADING DATASETS
b_cancer = datasets.load_breast_cancer()
load_b_cancer(b_cancer)
load_mpg()
load_fire()
load_student()

