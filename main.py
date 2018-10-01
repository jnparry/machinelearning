import csv

from sklearn import datasets
from sklearn.model_selection import train_test_split

from HardCodedClassifier import HardCodedClassifier

# K is the number of nearest neighbors
k = 1
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=20)


def main():
    # Show the data (the attributes of each instance)
    # print(iris.data)

    # Show the target values (in numeric format) of each instance
    # print(iris.target)

    # Show the actual target names that correspond to each number
    # print(iris.target_names)

    # print "\nX Train: \n", X_train
    # print "\nX Test: \n", X_test
    # print "\nY Train: \n", y_train
    # print "\nY Test: \n", y_test

    # Naive Bayes algorithm implementation GaussianNB to train a model
    # classifier = GaussianNB()

    # classifier = HardCodedClassifier()
    # model = classifier.fit(X_train, y_train)

    classifier = HardCodedClassifier(k)
    model = classifier.fit(X_train, y_train)

    targets_predicted = model.predict(X_test)

    items_correct = 0
    not_correct = 0

    # Compare the predictions with the answers
    for i in range(len(targets_predicted)):
        if targets_predicted[i] == y_test[i]:
            items_correct += 1
        else:
            not_correct += 1

    total_items = items_correct + not_correct

    # Calculating and displaying the amount correct
    print "\n\nMachine got ", items_correct, " correct out of ", total_items, ". Accuracy is at ", \
        100.00 * items_correct / total_items, "%.\n"


    # # Using a text file and retrieving data from it
    # with open('C:\Users\Jordan\Desktop\PycharmProjects\prove01\employee_brithday.txt') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     line_count = 0
    #
    #     # Loop
    #     for row in csv_reader:
    #         if line_count == 0:
    #             print "Column names are", row
    #             line_count += 1
    #         else:
    #             print "\t", row[0], "works in the", row[1], "department, and was born in", row[2]
    #             line_count += 1
    #
    #     # Tells us how many lines were used
    #     print "Processed", line_count, "lines."


if __name__ == "__main__":
    main()
