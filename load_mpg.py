from StringIO import StringIO

import numpy
import pandas as pd
import numpy as np
from sklearn import preprocessing


def load_mpg():
    # read in files and add temp column
    auto_mpg_file = pd.read_csv("auto_mpg_data.csv")
    auto_mpg_file.columns = ['temp']

    # reformat csv file
    formatted_data = ""
    for i in range(len(auto_mpg_file)):
        end = False
        split = auto_mpg_file['temp'][i].split()

        for j in range(len(split)):
            formatted_data += split[j]
            if split[j][0] == "\"":
                end = True

            if end:
                formatted_data += " "
            else:
                formatted_data += ";"
        formatted_data += "\n"

    # use the created string and convert to csv
    temp_data = StringIO("""MPG;Cylinders;Displacement;Horsepower;Weight;Acceleration;ModelYear;Origin;Name
        {0}
        """.format(formatted_data))
    mpg_file = pd.read_csv(temp_data, sep=";")

    # drop rows with missing data
    rows_with_missing_data = 0
    for row in mpg_file:
        for i in range(len(mpg_file)):
            if mpg_file[row][i] == '?':
                mpg_file[row][i] = numpy.NaN
                rows_with_missing_data += 1

    mpg_file.dropna(inplace=True)

    # label encode
    le = preprocessing.LabelEncoder()

    for row in mpg_file:
        le.fit(mpg_file[row])
        # print(list(le.classes_))
        mpg_file[row] = (le.transform(mpg_file[row]))

    # return two numpy arrays - the data and the target
    return np.array(mpg_file.loc[:, 'Cylinders':'Name']), np.array(mpg_file['MPG'])