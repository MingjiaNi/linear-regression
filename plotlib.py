import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_loader import build_dataset
import pandas as pd


def plot_dataset(data):

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data['x1'], data['x2'], data['f(X)'], c='b', marker='o')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(X)')
    ax.set_title('Visualizing training data')

    plt.show()


if __name__ == '__main__':
    dataset = build_dataset('./regression-dataset')
    plot_dataset(pd.concat(dataset))
