import numpy as np
from data_loader import build_dataset
import pandas as pd


def get_linear_regression_weights(dataset):
    """
    :param dataset: contains attribute and f-values
    :return: the weights vector for the underlying linear system of equations
    """
    N = len(dataset)
    A = np.zeros((3, 3), dtype=int)
    b = np.zeros(3, dtype=float)

    for index, row in dataset.iterrows():
        x = row.values[:-1]
        x_dash = np.append(1, x)
        x_dash.reshape((3, 1))
        b = np.add(b, np.multiply(row.values[-1], x_dash))
        print(np.matmul(x_dash, x_dash.reshape((1, 3))))
        #b += np.multiply(row.values[-1], x_dash)
        #print(x_dash)
        #print(row.values[-1])

    print(b)


if __name__ == '__main__':
    dataset = build_dataset('./regression-dataset')
    get_linear_regression_weights(pd.concat(dataset))
