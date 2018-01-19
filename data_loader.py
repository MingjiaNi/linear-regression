import pandas as pd
import numpy as np

class DataLoader:
    """
    Class that helps in loading dataset. Assumes attributes and labels are in separate file
    """

    num_files = 10

    @classmethod
    def load_dataset(cls, attributes_file, labels_file):
        """
        :param attributes_file: file path of attributes
        :param labels_file: file path of corresponding labels
        :return a 2-tuple of dataframes of attributes and labels

        Loads data from single (attributes, labels) file pair
        """

        # Load attributes file and value file
        df_attributes = pd.read_csv(attributes_file, header=None, dtype=int)
        df_labels = pd.read_csv(labels_file, header=None, dtype=float)

        if len(df_attributes) != len(df_labels):
            raise ValueError('count mismatch in attributes and labels file')

        return df_attributes, df_labels

    @classmethod
    def load_full_dataset(cls, dataset_root_directory):
        """
        :param dataset_root_directory: root directory containing all dataset files
        :return a 2-tuple of dataframes of attributes and labels

        Loads data from a collection of files
        """

        attributes_group = []
        labels_group = []
        for i in range(1, DataLoader.num_files + 1):
            attributes, labels = DataLoader.load_dataset(
                                    dataset_root_directory + '/fData' + str(i) + '.csv',
                                    dataset_root_directory + '/fLabels' + str(i) + '.csv'
                                    )
            attributes_group.append(attributes)
            labels_group.append(labels)

        return pd.concat(attributes_group, ignore_index=True), pd.concat(labels_group, ignore_index=True)

    @classmethod
    def load_with_test_data(cls, dataset_root_directory, split_ratio=0.1):

        if not 0 < split_ratio < 1:
            raise ValueError('Split ratio should be in (0,1)')

        test_attrs, test_labels = DataLoader.load_dataset(
                                      dataset_root_directory + '/fData1.csv',
                                      dataset_root_directory + '/fLabels1.csv')
        attributes_group = []
        labels_group = []
        for i in range(2, DataLoader.num_files + 1):
            attributes, labels = DataLoader.load_dataset(
                dataset_root_directory + '/fData' + str(i) + '.csv',
                dataset_root_directory + '/fLabels' + str(i) + '.csv'
            )
            attributes_group.append(attributes)
            labels_group.append(labels)

        train_attrs = pd.concat(attributes_group, ignore_index=True)
        train_labels = pd.concat(labels_group, ignore_index=True)

        return train_attrs, train_labels, test_attrs, test_labels


if __name__ == '__main__':

    part_train_attrs, part_train_labels = DataLoader.load_dataset(
                                              'regression-dataset/fData1.csv',
                                              'regression-dataset/fLabels1.csv'
                                          )
    print('='*5 + 'Training data from single file' + '='*5)
    print('Attributes = ', part_train_attrs.shape)
    print('Labels = ', part_train_labels.shape)
    print()

    train_attrs, train_labels = DataLoader.load_full_dataset('./regression-dataset')
    print('=' * 15 + 'Full dataset' + '=' * 15)
    print('Attributes = ', train_attrs.shape)
    print('Labels = ', train_labels.shape)
    print()

    train_attrs, train_labels, test_attrs, test_labels = DataLoader.load_with_test_data(
                                                            './regression-dataset',
                                                            split_ratio=0.2)
    print('=' * 5 + 'Train-Test split' + '=' * 5)
    print('Train Attributes = ', train_attrs.shape)
    print('Train Labels = ', train_labels.shape)
    print('Test Attributes = ', test_attrs.shape)
    print('Test Labels = ', test_labels.shape)
