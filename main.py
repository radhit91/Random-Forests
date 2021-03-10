import datetime

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

np.random.seed(1)

# data comes from https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results
data = np.genfromtxt('dataset/dota2Train.csv', delimiter=',')
test_data = np.genfromtxt('dataset/dota2Test.csv', delimiter=',')

np.random.shuffle(data)
#
X_train, Y_train = data[:, 1:], data[:, 0]
X_test, Y_test = test_data[:, 1:], test_data[:, 0]


def standardize(X):
    return (X - np.mean(X)) / np.std(X)


def select_features(X, n=30):
    print("Select features with PCA: n=%d" % n)
    pca = PCA(n_components=n)
    pca.fit(standardize(X))
    n_pcs = pca.components_.shape[0]
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    print(most_important)
    return X[:, np.unique(most_important)]


def select_topN_herros(X, Y, n=30):
    print("Select features with top hero: n=%d" % n)
    heros = X[:,
            3:]  # indicator for a hero, Value of 1 indicates that a player from team '1' played as that hero and '-1' for the other team

    Y_1 = np.reshape(Y, (-1, 1))
    # Y_neg1 = np.reshape(Y,(-1,1))

    result = Y_1 - heros  # if the result is 0, it means team 1 win the game by using selected heroes
    # result_neg1 = Y_neg1- heros #if the result is 0, it means team -1 win the game by using selected heroes

    print(np.count_nonzero(result == 0, axis=0))  # calculate the total win times of every champs in team 1
    # [3634 5552 ....] 3634 means champ index 0 win 3634 games out of 92650 games in team 1

    # choose top 22 champs(index) win games
    idx = np.argsort(np.count_nonzero(result, axis=0))[-n:]
    return np.concatenate((X_train[:, :3], X_train[:, np.array(idx) + 3]), axis=1)


def train_on_NN():
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    hidden_layers = (50, 30, 15, 5)
    alpha = 0.1
    print("train on NN with hidden layer %s alpha=%f" % (hidden_layers, alpha))
    print("Start: %s" % datetime.datetime.now().ctime())
    selected_train = select_features(X_train, n=100)
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(alpha=alpha, hidden_layer_sizes=(50, 30, 15, 5), random_state=1, max_iter=10000))
    res = cross_val_score(clf, selected_train, Y_train, cv=10)
    print("Finish: %s" % datetime.datetime.now().ctime())
    print(res)

    return np.mean(res)


def train_on_RF():
    print("selected feature with PCA")

    selected_train = select_features(X_train, n=100)
    print(selected_train.shape)
    print("Start: %s" % datetime.datetime.now().ctime())
    clf = RandomForestClassifier(random_state=0)
    res = cross_val_score(clf, selected_train, Y_train, cv=10)
    print("Finish: %s" % datetime.datetime.now().ctime())
    print(res)

    return np.mean(res)


def train_on_SVM():
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    print("train on SVM")
    print("Start: %s" % datetime.datetime.now().ctime())
    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, max_iter=10000))
    res = cross_val_score(clf, X_train, Y_train, cv=10)
    print("Finish: %s" % datetime.datetime.now().ctime())
    print(res)

    return np.mean(res)


if __name__ == '__main__':
    train_on_NN()
