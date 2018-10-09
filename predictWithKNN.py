from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def knn_predict(data, target, k):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=10)

    # Is the OTS KNN - fit and predict
    neighbors = KNeighborsClassifier(n_neighbors=k)
    neighbors.fit(X_train, y_train)
    targets_predicted = neighbors.predict(X_test)

    # Compare the predictions with the answers
    items_correct = 0
    total_items = len(targets_predicted)

    for i in range(len(y_test)):
        if targets_predicted[i] == y_test[i]:
            items_correct += 1

    # Calculating and displaying the amount correct
    print "\n\nMachine got ", items_correct, " correct out of ", total_items, ". Accuracy is at ", \
        100.00 * items_correct / total_items, "%.\n"
