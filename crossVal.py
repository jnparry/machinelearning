from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def cross_val(X, y, k):
    # Applying K Fold Cross Validation and initializing classifier
    k_fold = KFold(n_splits=10, shuffle=True, random_state=7)
    classifier = KNeighborsClassifier(n_neighbors=k, metric='minkowski')

    # Prediction and Accuracy Results
    y_pred = cross_val_predict(classifier, X, y, cv=k_fold, n_jobs=1)
    accuracy_score = cross_val_score(classifier, X, y, cv=k_fold, n_jobs=1).mean()

    print "Cross validation accuracy is", accuracy_score * 100
