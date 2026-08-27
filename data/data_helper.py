from random import randint


def gen_deterministic_data(n):
    # return [[3, 5, 2]], [10]
    X = [[randint(1, 10) for _ in range(3)] for _ in range(n)]
    y = [(5*x[0])^2 + (x[1])^3 + 3*x[2] for x in X]
    return X, y


def split_data(X, y):
    index_cutoff = round(len(X) * 0.8)
    X_train = X[:index_cutoff]
    X_test = X[index_cutoff:]
    y_train = y[:index_cutoff]
    y_test = y[index_cutoff:]
    return X_train, X_test, y_train, y_test