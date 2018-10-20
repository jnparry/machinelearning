import numpy
import pandas as pd
import numpy as np
from sklearn import preprocessing


def load_aut():
    # read in file
    aut_file = pd.read_csv("Autism-Adult-Data.csv")

    # rename columns
    aut_file.columns = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Age', 'Gender', 'Ethnicity',
                        'Jaundice', 'Family_PDD', 'Country', 'Used_App', 'Screening_Score', 'Unknown', 'Person_Testing',
                        'Target']

    # drop rows with missing data
    rows_with_missing_data = 0
    for row in aut_file:
        for i in range(len(aut_file)):
            if aut_file[row][i] == '?':
                aut_file[row][i] = numpy.NaN
                rows_with_missing_data += 1

    aut_file.dropna(inplace=True)

    # label encode
    le = preprocessing.LabelEncoder()

    for row in aut_file:
        le.fit(aut_file[row])
        # print(list(le.classes_))
        aut_file[row] = (le.transform(aut_file[row]))

    # return two numpy arrays - the data and the target
    return np.array(aut_file.loc[:, 'Q1':'Person_Testing']), np.array(aut_file['Target'])
