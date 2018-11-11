from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def load_b_cancer(my_file):
    X_train, X_test, y_train, y_test = train_test_split(my_file.data, my_file.target, test_size=0.3, random_state=42)

    # Normalizing
    test_train = [X_train, X_test]
    for i in range(len(test_train)):
        test_scaler = StandardScaler(False)
        test_scaler.fit(test_train[i])
        test_scaler.transform(test_train[i])

    # Neural Network
    clf = MLPClassifier(solver='lbfgs', max_iter=10, activation='tanh', hidden_layer_sizes=(5, 2), random_state=55)
    clf.fit(X_train, y_train)

    # Compare the predictions with the answers
    targets_predicted = clf.predict(X_test)
    items_correct = 0
    not_correct = 0

    for i in range(len(targets_predicted)):
        if targets_predicted[i] == y_test[i]:
            items_correct += 1
        else:
            not_correct += 1

    total_items = items_correct + not_correct

    # Calculating and displaying the amount correct
    print "\n\nCancer prediciton got", items_correct, "correct out of", total_items, ". Accuracy is at", \
        round(100.00 * items_correct / total_items, 1), "%."
