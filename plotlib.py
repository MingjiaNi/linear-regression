import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_loader import DataLoader


def plot_dataset(attributes, labels):
    """
    :param attributes: attributes of all data-points
    :param labels: corresponding true labels

    For 3D visualizing the dataset
    """

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(
        attributes[attributes.columns[0]],
        attributes[attributes.columns[1]],
        labels[labels.columns[0]],
        label='training data',
        c='b', marker='o')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(X)')
    ax.set_title('Visualizing dataset')
    ax.legend()

    plt.show()


def plot_dataset_with_predictions(train_attrs, train_labels, test_attrs, test_labels, test_predictions):
    """
    :param train_attrs: attributes of all data-points in training set
    :param train_labels: corresponding true labels of training set
    :param test_attrs: attributes of all data-points in test set
    :param test_labels: corresponding true labels of test set
    :param test_predictions: model predictions of test set labels

    For 3D visualizing the predicted data
    """

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(
        train_attrs[train_attrs.columns[0]],
        train_attrs[train_attrs.columns[1]],
        train_labels[train_labels.columns[0]],
        label='training data',
        c='b', marker='o')
    ax.scatter(
        test_attrs[test_attrs.columns[0]],
        test_attrs[test_attrs.columns[1]],
        test_labels[test_labels.columns[0]],
        label='test data true value',
        c='r', marker='o')
    ax.scatter(
        test_attrs[test_attrs.columns[0]],
        test_attrs[test_attrs.columns[1]],
        test_predictions[test_predictions.columns[0]],
        label='test data predictions',
        c='g', marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(X)')
    ax.set_title('Predictions')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    train_attrs, train_labels = DataLoader.load_full_dataset('./regression-dataset')
    plot_dataset(train_attrs, train_labels)
