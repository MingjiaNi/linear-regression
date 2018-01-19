import numpy as np
from data_loader import DataLoader
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self):
        self.weights = None
        self.best_lambda = 0  # The best lambda is learned by the system
        self.best_E_RMS = 0

        # For plotting
        self.lambda_values = []
        self.E_RMS_values = []

    def learn_with_regularization(self, train_attrs, train_labels, _lambda=0):

        if _lambda < 0:
            raise ValueError('regularization_weight should be non-negative')

        if len(train_attrs) != len(train_labels):
            raise ValueError('count mismatch in attributes and labels')

        # M is the no of attributes
        M = train_attrs.shape[1]

        regularization_terms = _lambda * np.identity(M + 1)
        A = np.zeros((M + 1, M + 1), dtype=float)
        b = np.zeros(M + 1, dtype=float)

        for index, row in train_attrs.iterrows():
            X = row.values
            f_X = train_labels.iat[index, 0]
            X_dash = np.append(1, X)
            b = np.add(b, np.multiply(f_X, X_dash))
            A = np.add(A, np.multiply(X_dash, X_dash.reshape((M + 1, 1))))

        A = np.add(A, regularization_terms)
        self.weights = np.linalg.solve(A, b)

    def learn(self, train_attrs, train_labels, verbose=False):
        learned_lambda = self.get_best_lambda(train_attrs, train_labels, verbose)[0]
        self.learn_with_regularization(train_attrs, train_labels, _lambda=learned_lambda)

    def predict_label(self, X):
        return self.weights[0] + X[0] * self.weights[1] + X[1] * self.weights[2]

    def predict(self, test_attrs, true_values=None):

        if not true_values.empty:
            if len(test_attrs) != len(true_values):
                raise ValueError('count mismatch in attributes and labels')
            E_RMS = 0.0
            N = len(test_attrs)

        predicted_values = []
        for index, row in test_attrs.iterrows():
            X = row.values
            predicted_value = self.predict_label(X)
            predicted_values.append(predicted_value)
            if not true_values.empty:
                true_value = true_values.iat[index, 0]
                E_RMS += (true_value - predicted_value) ** 2

        predicted_values = pd.DataFrame(np.array(predicted_values))

        if not true_values.empty:
            E_RMS = E_RMS / N
            E_RMS = sqrt(E_RMS)
            return predicted_values, E_RMS
        else:
            return predicted_values, None

    def k_fold_cross_validation(self, attributes, labels, k=10,  _lambda=0):

        N = len(attributes)
        if N != len(labels):
            raise ValueError('count mismatch in attributes and labels')

        subset_size = N // k
        start = 0
        end = 0
        avg_E_RMS = 0
        for i in range(1, k + 1):
            start = end
            end = i * subset_size
            attrs_splits = np.split(attributes, [start, end])
            labels_splits = np.split(labels, [start, end])
            test_attrs = attrs_splits[1]
            test_labels = labels_splits[1]
            test_attrs = test_attrs.reset_index(drop=True)
            test_labels = test_labels.reset_index(drop=True)
            train_attrs = pd.concat([attrs_splits[0], attrs_splits[2]], ignore_index=True)
            train_labels = pd.concat([labels_splits[0], labels_splits[2]], ignore_index=True)
            self.learn_with_regularization(train_attrs, train_labels, _lambda=_lambda)
            E_RMS = self.predict(test_attrs, true_values=test_labels)[1]
            avg_E_RMS += E_RMS

        return avg_E_RMS / 10

    def get_best_lambda(self, attributes, values, verbose=False):

        self.lambda_values = []
        self.E_RMS_values = []
        _lambda = 0.0
        while _lambda < 4.1:
            E_RMS = self.k_fold_cross_validation(attributes, values, k=10, _lambda=_lambda)
            if verbose:
                print('At lambda = %f E_ERMS = %f' % (_lambda, E_RMS))
            self.lambda_values.append(round(_lambda, 1))
            self.E_RMS_values.append(E_RMS)
            _lambda += 0.1

        self.best_E_RMS = min(self.E_RMS_values)
        self.best_lambda = self.lambda_values[self.E_RMS_values.index(self.best_E_RMS)]

        return self.best_lambda, self.best_E_RMS

    def summary(self):
        print('\n')
        print('='*10 + 'Model Summary' + '='*10)
        print('Model Weights =', end=' ')
        print(self.weights)
        print('Learned Lambda = %.2f' % self.best_lambda)
        print('Least Root Mean Square Error = %f' % self.best_E_RMS)

    def plot_E_RMS(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.lambda_values, self.E_RMS_values, '.-')

        ymin = self.best_E_RMS
        xmin = self.best_lambda
        ax.annotate(
            'min at lambda=%f, E_RMS=%f ' % (xmin, ymin),
            xy=(xmin, ymin), xytext=(xmin - 1, ymin + 0.0005),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1),
            )
        ax.scatter([xmin],[ymin], c='r')
        plt.xlabel('lambda')
        plt.ylabel('E_RMS')
        plt.title('E_RMS vs lambda')
        plt.show()


if __name__ == '__main__':

    attributes, labels = DataLoader.load_full_dataset('./regression-dataset')

    model = LinearRegression()
    model.learn(attributes, labels, verbose=True)
    model.summary()
    model.plot_E_RMS()

