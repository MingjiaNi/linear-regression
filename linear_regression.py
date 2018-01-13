import numpy as np
from data_loader import build_dataset
import pandas as pd
from plotlib import plot_dataset, plot_3D, plot_predictions
from math import sqrt
import matplotlib.pyplot as plt


def get_linear_regression_weights(dataset, regularization_weight=0):
    """
    :param dataset: contains attribute and f-values
    :return: the weights vector for the underlying linear system of equations
    """

    if regularization_weight < 0:
        raise ValueError('regularization_weight should be non-negative')

    # M is the no of attributes
    M = dataset.shape[1] - 1

    A = np.zeros((M + 1, M + 1), dtype=float)
    b = np.zeros(M + 1, dtype=float)

    for index, row in dataset.iterrows():
        X = row.values[:-1]
        f_X = row.values[-1]
        X_dash = np.append(1, X)
        b = np.add(b, np.multiply(f_X, X_dash))
        A = np.add(A, np.multiply(X_dash, X_dash.reshape((M+1, 1))))

    lambda_identity = regularization_weight * np.identity(M + 1)
    A = np.add(A, lambda_identity)

    #print(A)

    w = np.linalg.solve(A, b)
    #print(w)

    return w


def compute_y(X, weights):
    return weights[0] + X[0] * weights[1] + X[1] * weights[2]


def predict(test_set, weights):

    predictions = []
    E_RMS = 0.0
    N = len(test_set)
    for index, row in test_set.iterrows():
        X = row.values[:-1]
        true_value = row.values[-1]
        predicted_value = compute_y(X, weights)
        predictions.append(predicted_value)

        E_RMS += (true_value - predicted_value)**2

    E_RMS = E_RMS / N
    E_RMS = sqrt(E_RMS)

    return predictions, E_RMS


def ten_fold_cross_validation(full_dataset, regularization_weight=0):

    avg_E_RMS = 0
    for i in range(10):
        test_set = full_dataset.pop(0)
        weights = get_linear_regression_weights(pd.concat(full_dataset), regularization_weight)
        E_RMS = predict(test_set, weights)[1]
        avg_E_RMS += E_RMS
        full_dataset.append(test_set)

    return avg_E_RMS / 10


lambda_values = []
E_RMS_values = []


def get_best_lambda(full_dataset):

    regularization_weight = 0.0
    while regularization_weight < 4.1:
        E_RMS = ten_fold_cross_validation(full_dataset, regularization_weight=regularization_weight)
        print('At lambda = %f E_ERMS = %f' % (regularization_weight, E_RMS))
        lambda_values.append(round(regularization_weight, 1))
        E_RMS_values.append(E_RMS)
        regularization_weight += 0.1

    best_E_RMS = min(E_RMS_values)
    best_lambda = lambda_values[E_RMS_values.index(best_E_RMS)]

    return best_lambda, best_E_RMS


def plot_E_RMS():

    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(lambda_values, E_RMS_values, '.-')

    ymin = min(E_RMS_values)
    xpos = E_RMS_values.index(ymin)
    xmin = lambda_values[xpos]
    print(lambda_values)
    print('Selected point', xmin, ymin, xpos)
    ax.annotate('min at lambda=%f, E_RMS=%f ' % (xmin, ymin), xy=(xmin, ymin), xytext=(xmin - 1, ymin + 0.0005),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                )
    ax.scatter([xmin],[ymin], c='r')
    plt.xlabel('lambda')
    plt.ylabel('E_RMS')
    plt.title('E_RMS vs lambda')
    plt.show()
    plt.savefig("output.png")


if __name__ == '__main__':
    dataset = build_dataset('./regression-dataset')
    train_set = dataset[:-1]
    cross_validation = dataset[-1]

    weights = get_linear_regression_weights(pd.concat(train_set))
    predictions, E_RMS = predict(cross_validation, weights)
    print(predictions)
    print(cross_validation['f(X)'])
    print('Root Mean Square Error = %.2f' % E_RMS)

    print(ten_fold_cross_validation(dataset, regularization_weight=0.5))
    #plot_dataset(pd.concat(train_set))

    #plot_predictions(pd.concat(train_set), cross_validation, predictions)
    best_lambda, best_E_RMS = get_best_lambda(dataset)
    print('Best lambda = %f with E_RMS = %.2f' % (best_lambda, best_E_RMS))

    plot_E_RMS()


