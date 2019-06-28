import os
import pandas as pd
import numpy as np


def concat_test():
    dir = 'feature_Data_test'
    files = os.listdir(dir)
    os.chdir(dir)
    data = pd.DataFrame()
    for file in files:
        csv = pd.read_csv(file)
        csv.drop(['time'], inplace=True, axis=1)
        print(csv.shape)
        data = pd.concat((data, csv), axis=1)
    print(data.shape)
    data.to_csv('../test/feature.csv', index=False)


def concat_train():
    dir = 'feature_Data_data_filter'
    files = os.listdir(dir)
    os.chdir(dir)
    data = pd.DataFrame()
    for file in files:
        csv = pd.read_csv(file)
        csv.drop(['time'], inplace=True, axis=1)
        print(csv.shape)
        data = pd.concat((data, csv), axis=1)
    print(data.shape)
    data.to_csv('../data/feature_filter.csv', index=False)


if __name__ == '__main__':
    concat_train()
