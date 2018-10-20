import os
import pandas

from sklearn.model_selection import cross_val_score
from create_visual import create_visual
from load_mpg import load_mpg
from load_car import load_car
from sklearn import tree, preprocessing
from sklearn import datasets

# car_data.shape - Function that I keep deleting and then re-looking up...
# Issues with graphviz - solved by adding this line. Who knows why...
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'

# LOADING DATASETS
car_file = pandas.read_csv("car_data.csv")
iris = datasets.load_iris()
mpg_data, mpg_target = load_mpg()


# ONE HOT ENCODING
le = preprocessing.LabelEncoder()
car_file = car_file.apply(le.fit_transform)
enc = preprocessing.OneHotEncoder()
enc.fit(car_file)
onehotlabels = enc.transform(car_file).toarray()

# Separating into target and data
car_target = onehotlabels[:, (len(onehotlabels[0]) - 1)]
car_data = onehotlabels[:, 0:(len(onehotlabels[0]) - 2)]

# Car decision tree
car_tree = tree.DecisionTreeClassifier()
car_tree = car_tree.fit(car_data, car_target)
create_visual(car_tree, "car_tree.png")

# REGULAR CATEGORICAL TREE - LABEL ENCODING
car_data_2, car_target_2 = load_car()
car_tree_2 = tree.DecisionTreeClassifier(max_leaf_nodes=20)
car_tree_2 = car_tree_2.fit(car_data_2, car_target_2)
create_visual(car_tree_2, "car_tree_2.png")

# the data is all the other attributes, the target is the mpg
mpg_tree = tree.DecisionTreeRegressor(random_state=0, max_depth=5)
mpg_tree = mpg_tree.fit(mpg_data, mpg_target)
# print cross_val_score(mpg_tree, mpg_data, mpg_target, cv=10)
create_visual(mpg_tree, "mpg_tree.png")

# BINNING
# Iris decision tree
iris_tree = tree.DecisionTreeClassifier()
iris_tree = iris_tree.fit(iris.data, iris.target)

rowAverage = []

# LOOPING THROUGH EACH COLUMN TO GET THE AVERAGE
for j in range(4):
    sum = 0
    for i in range(len(iris.data)):
        sum += iris.data[i][j]
    rowAverage.append(sum / len(iris.data))

# ONLY TWO BINS, DIVIDING LINE IS THE AVERAGE
for i in range(len(iris.data)):
    for j in range(4):
        if iris.data[i][j] < rowAverage[j]:
            iris.data[i][j] = 0
        else:
            iris.data[i][j] = 1

create_visual(iris_tree, "iris_tree.png")
