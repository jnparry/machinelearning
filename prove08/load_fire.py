import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

# np.set_printoptions(suppress=True)


def load_fire():
    # read in files and add temp column
    data = pd.read_csv("forestfires.csv")
    data.columns = ['x', 'y', 'month', 'day', 'ffmc', 'dmc', 'dc', 'isi', 'temp', 'rh', 'wind', 'rain', 'area']

    # label encode
    le = preprocessing.LabelEncoder()

    for row in data:
            le.fit(data[row])
            data[row] = (le.transform(data[row]))

    # return two numpy arrays - the data and the target
    my_data = np.array(data.loc[:, 'x':'rain'])
    target = np.array(data['area'])

    # Feature Selection
    my_data = SelectKBest(chi2, k=11).fit_transform(my_data, target)

    X_train, X_test, y_train, y_test = train_test_split(my_data, target, test_size=0.3, random_state=20)

    # Neural Network
    rgr = MLPRegressor(random_state=55, solver='adam', activation='identity', learning_rate='constant')
    rgr.fit(X_train, y_train)
    targets_predicted = rgr.predict(X_test)

    # Print the score
    print "Forest fire regression score is", rgr.score(X_test, y_test)
