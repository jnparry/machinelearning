from random import randint
from sklearn.preprocessing import StandardScaler

num_attributes = 4

class HardCodedModel:
    def __init__(self, k, data, targets):
        # Scales the training_data
        scaler = StandardScaler(False)
        scaler.fit(data)
        scaler.transform(data)

        self.k = k
        self.train_x = data
        self.train_y = targets

    def predict(self, test_data):
        # Scales the test_data
        test_scaler = StandardScaler(False)
        test_scaler.fit(test_data)
        test_scaler.transform(test_data)

        # We will return this array of predictions
        prediction = []

        # Main loop - Loops through every test item in order to guess
        for i in range(len(test_data)):
            # Reset the distances for every test data with all the training data
            distances = []

            # Loop through the training data
            for j in range(len(self.train_x)):
                # Reset the distance
                distance = 0

                # Loop through the four attributes to add the distance
                for k in range(num_attributes):
                    distance += (test_data[i][k] - self.train_x[j][k]) ** 2

                # After summing up the distances from the 4 attributes, append to distances
                distances.append(distance)

            # Reset the number of times the flower is the closest.
            setosa = 0
            versicolor = 0
            virginica = 0

            # Looping through the number of closest neighbors, k
            for l in range(self.k):
                smallest_index = 0

                # Loop through the distances and compare them to get the smallest distance
                for m in range(1, (len(distances))):
                    if distances[smallest_index] > distances[m]:
                        smallest_index = m

                # Which ever target is connected with the smallest distance, increment that variable
                if self.train_y[smallest_index] == 0:
                    setosa += 1
                elif self.train_y[smallest_index] == 1:
                    versicolor += 1
                elif self.train_y[smallest_index] == 2:
                    virginica += 1
                else:
                    print "Error."

                # Since we'll be looping through k times, discard the smallest so we can find the next smallest
                distances.pop(smallest_index)

            # Once we've gotten all the k closest neighbors tallied, make a prediction based on that
            if (setosa > versicolor) & (setosa > virginica):
                prediction.append(0)
            elif (versicolor > setosa) & (versicolor > virginica):
                prediction.append(1)
            elif (virginica > versicolor) & (virginica > setosa):
                prediction.append(2)
            elif (setosa == versicolor) & (versicolor == virginica):
                prediction.append(randint(0, 2))
            elif setosa == versicolor:
                prediction.append(randint(0, 1))
            elif versicolor == virginica:
                prediction.append(randint(1, 2))
            elif setosa == virginica:
                prediction.append(0)
            else:
                print "Error."

        return prediction
