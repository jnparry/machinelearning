from StringIO import StringIO

import numpy
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor


def load_mpg():
    # read in files and add temp column
    mpg_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    mpg_file = pd.read_csv(mpg_data, delim_whitespace=True)
    mpg_file.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'mYear', 'origin',
                   'carName']

    # label encode
    le = preprocessing.LabelEncoder()

    for row in mpg_file:
        if row != 'mpg':
            le.fit(mpg_file[row])
            # print(list(le.classes_))
            mpg_file[row] = (le.transform(mpg_file[row]))

    # return two numpy arrays - the data and the target
    mpg_data = np.array(mpg_file.loc[:, 'cylinders':'carName'])
    mpg_target = np.array(mpg_file['mpg'])
    mpgX_train, mpgX_test, mpgy_train, mpgy_test = train_test_split(mpg_data, mpg_target, test_size=0.3, random_state=20)

    # Neural Network
    rgr = MLPRegressor(random_state=55)
    rgr.fit(mpgX_train, mpgy_train)
    targets_predicted = rgr.predict(mpgX_test)

    # Print the score
    print "MPG regression score is", rgr.score(mpgX_test, mpgy_test)
