import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor


def load_student():
    # read in files and add temp column
    data = pd.read_csv("student-por.csv", sep=';')
    data.columns = ['school', 'sex', 'age', 'address', 'famsize', 'pstatus', 'medu', 'fedu', 'mjob', 'fjob', 'reason',
                    'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities',
                    'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'dalc', 'walc',
                    'health', 'absences', 'remove1', 'remove2', 'final_grade']

    # label encode
    le = preprocessing.LabelEncoder()

    for row in data:
        le.fit(data[row])
        data[row] = (le.transform(data[row]))

    # return two numpy arrays - the data and the target
    my_data = np.array(data.loc[:, 'school':'absences'])
    target = np.array(data['final_grade'])
    X_train, X_test, y_train, y_test = train_test_split(my_data, target, test_size=0.3, random_state=50)

    # Neural Network
    rgr = MLPRegressor(random_state=10)
    rgr.fit(X_train, y_train)

    # Print the score
    print "Student performance regression score is", rgr.score(X_test, y_test)
