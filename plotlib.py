import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_loader import build_dataset
import pandas as pd


def plot_3D(x1, x2, y, title):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x1, x2, y, c='b', marker='o')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(X)')
    ax.set_title(title)

    plt.show()


def plot_dataset(data):

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data['x1'], data['x2'], data['f(X)'], c='b', marker='o')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(X)')
    ax.set_title('Visualizing training data')

    plt.show()


def plot_predictions(train_data, test_data, predictions):

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(train_data['x1'], train_data['x2'], train_data['f(X)'], c='b', marker='o')
    ax.scatter(test_data['x1'], test_data['x2'], test_data['f(X)'], c='r', marker='o')
    ax.scatter(test_data['x1'], test_data['x2'], predictions, c='g', marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(X)')
    ax.set_title('Predictions')

    plt.show()



if __name__ == '__main__':
    dataset = build_dataset('./regression-dataset')
    plot_dataset(pd.concat(dataset))
