import pandas as pd
import numpy as np
from sklearn import preprocessing


def load_car():
    # read in file
    car_file = pd.read_csv("car_data.csv")

    # rename columns
    car_file.columns = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_Boot', 'Safety', 'Target']

    # label encode
    le = preprocessing.LabelEncoder()

    for row in car_file:
        le.fit(car_file[row])
        car_file[row] = (le.transform(car_file[row]))

    # return two numpy arrays - the data and the target
    return np.array(car_file.loc[:, 'Buying':'Safety']), np.array(car_file['Target'])

