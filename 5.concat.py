import os
import pandas as pd
import numpy as np


def concat_test():
    dir = 'feature_unsorted_test'
    files = os.listdir(dir)
    os.chdir(dir)
    data = pd.DataFrame()
    a = ['acc_x_fft.csv', 'acc_y_fft.csv', 'acc_xy_fft.csv', 'acc_xyz_fft.csv', 'acc_z_fft.csv',
         'g_x_fft.csv', 'g_y_fft.csv', 'g_z_fft.csv',
         'gy_x_fft.csv', 'gy_y_fft.csv', 'gy_z_fft.csv',
         'l_x_fft.csv', 'l_y_fft.csv', 'l_z_fft.csv',
         'm_x_fft.csv', 'm_y_fft.csv', 'm_z_fft.csv',
         'magnetic_fft.csv', 'o_w_fft.csv', 'o_x_fft.csv', 'o_y_fft.csv', 'o_z_fft.csv', 'pitch_fft.csv',
         'pressure_fft.csv',
         'roll_fft.csv', 'yaw_fft.csv']

    for file in files:
        # if file.find('fft') < 0:
            csv = pd.read_csv(file)
            csv.drop(['time'], inplace=True, axis=1)
            print(file, csv.shape)
            data = pd.concat((data, csv), axis=1)
    print(data.shape)
    data.to_csv('../test/feature_unsorted_60.csv', index=False)


def concat_train():
    dir = 'feature_unsorted_train'
    files = os.listdir(dir)
    os.chdir(dir)
    data = pd.DataFrame()
    a = ['acc_x_fft.csv', 'acc_y_fft.csv', 'acc_xy_fft.csv', 'acc_xyz_fft.csv', 'acc_z_fft.csv',
         'g_x_fft.csv', 'g_y_fft.csv', 'g_z_fft.csv',
         'gy_x_fft.csv', 'gy_y_fft.csv', 'gy_z_fft.csv',
         'l_x_fft.csv', 'l_y_fft.csv', 'l_z_fft.csv',
         'm_x_fft.csv', 'm_y_fft.csv', 'm_z_fft.csv',
         'magnetic_fft.csv', 'o_w_fft.csv', 'o_x_fft.csv', 'o_y_fft.csv', 'o_z_fft.csv', 'pitch_fft.csv',
         'pressure_fft.csv',
         'roll_fft.csv', 'yaw_fft.csv']
    for file in files:
        # if file.find('fft') < 0:
            csv = pd.read_csv(file)
            csv.drop(['time'], inplace=True, axis=1)
            print(file, csv.shape)
            data = pd.concat((data, csv), axis=1)
    print(data.shape)
    data.to_csv('../data/feature_unsorted_60.csv', index=False)


if __name__ == '__main__':
    # concat_train()
    concat_test()
