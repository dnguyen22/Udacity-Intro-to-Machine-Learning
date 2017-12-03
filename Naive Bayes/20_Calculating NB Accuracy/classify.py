def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of Naive Bayes classifier """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    return accuracy_score(labels_test, pred)
