import pandas as pd
import pprint


def build_ith_dataset(train_attributes_file, train_label_file):

    # Load attributes file and f-value file
    df_attributes = pd.read_csv(train_attributes_file, header=None, dtype=int)
    df_f_value = pd.read_csv(train_label_file, header=None, dtype=float)
    if len(df_attributes) != len(df_f_value):
        raise ValueError('count mismatch in attributes and target value file')

    column_names = ['x1', 'x2', 'f(X)']
    combined_df = pd.concat([df_attributes, df_f_value], axis=1)
    combined_df.columns = column_names

    return combined_df


def build_dataset(dataset_path, k=10):

    dataset = []
    for i in range(1, k + 1):
        ith_dataset = build_ith_dataset(dataset_path + '/fData' + str(i) + '.csv',
                                        dataset_path + '/fLabels' + str(i) + '.csv')
        dataset.append(ith_dataset)

    return dataset


if __name__ == '__main__':

    dataset_ith = build_ith_dataset('regression-dataset/fData1.csv', 'regression-dataset/fLabels1.csv')
    print('='*5 + 'Subset of training data' + '='*5)
    pprint.pprint(dataset_ith)
    print('\n\n')

    dataset = build_dataset('./regression-dataset')
    print('=' * 15 + 'Full dataset' + '=' * 15)
    print(pd.concat(dataset).describe())
