import os

import pandas as pd
import numpy as np


# 把txt文件合并并且转化成csv文件，并且对应上每条记录的label

def train():
    os.chdir(r'data/data')
    acc_x = pd.read_csv("Acc_x.txt", ' ', header=None)
    acc_y = pd.read_csv("Acc_y.txt", ' ', header=None)
    acc_z = pd.read_csv("Acc_z.txt", ' ', header=None)
    acc_x = pd.DataFrame(np.asarray(acc_x).flatten())
    acc_y = pd.DataFrame(np.asarray(acc_y).flatten())
    acc_z = pd.DataFrame(np.asarray(acc_z).flatten())

    acc = pd.concat((acc_x, acc_y, acc_z), axis=1)

    print(acc.shape)

    g_x = pd.read_csv("Gra_x.txt", ' ', header=None)
    g_y = pd.read_csv("Gra_y.txt", ' ', header=None)
    g_z = pd.read_csv("Gra_z.txt", ' ', header=None)
    g_x = pd.DataFrame(np.asarray(g_x).flatten())
    g_y = pd.DataFrame(np.asarray(g_y).flatten())
    g_z = pd.DataFrame(np.asarray(g_z).flatten())
    g = pd.concat((g_x, g_y, g_z), axis=1)

    print(g.shape)

    l_x = pd.read_csv("LAcc_x.txt", ' ', header=None)
    l_y = pd.read_csv("LAcc_y.txt", ' ', header=None)
    l_z = pd.read_csv("LAcc_z.txt", ' ', header=None)
    l_x = pd.DataFrame(np.asarray(l_x).flatten())
    l_y = pd.DataFrame(np.asarray(l_y).flatten())
    l_z = pd.DataFrame(np.asarray(l_z).flatten())
    l = pd.concat((l_x, l_y, l_z), axis=1)

    print(l.shape)

    m_x = pd.read_csv("Mag_x.txt", ' ', header=None)
    m_y = pd.read_csv("Mag_y.txt", ' ', header=None)
    m_z = pd.read_csv("Mag_z.txt", ' ', header=None)
    m_x = pd.DataFrame(np.asarray(m_x).flatten())
    m_y = pd.DataFrame(np.asarray(m_y).flatten())
    m_z = pd.DataFrame(np.asarray(m_z).flatten())
    m = pd.concat((m_x, m_y, m_z), axis=1)

    print(m.shape)

    o_w = pd.read_csv("Ori_w.txt", ' ', header=None)
    o_x = pd.read_csv("Ori_x.txt", ' ', header=None)
    o_y = pd.read_csv("Ori_y.txt", ' ', header=None)
    o_z = pd.read_csv("Ori_z.txt", ' ', header=None)
    o_x = pd.DataFrame(np.asarray(o_x).flatten())
    o_y = pd.DataFrame(np.asarray(o_y).flatten())
    o_z = pd.DataFrame(np.asarray(o_z).flatten())
    o_w = pd.DataFrame(np.asarray(o_w).flatten())
    o = pd.concat((o_w, o_x, o_y, o_z), axis=1)

    print(o.shape)

    gy_x = pd.read_csv("Gyr_x.txt", ' ', header=None)
    gy_y = pd.read_csv("Gyr_y.txt", ' ', header=None)
    gy_z = pd.read_csv("Gyr_z.txt", ' ', header=None)
    gy_x = pd.DataFrame(np.asarray(gy_x).flatten())
    gy_y = pd.DataFrame(np.asarray(gy_y).flatten())
    gy_z = pd.DataFrame(np.asarray(gy_z).flatten())
    gy = pd.concat((gy_x, gy_y, gy_z), axis=1)

    print(gy.shape)
    pressure = pd.read_csv("Pressure.txt", ' ', header=None)
    pressure = pd.DataFrame(np.asarray(pressure).flatten())
    print(pressure.shape)

    label = pd.read_csv("Label.txt", ' ', header=None)
    label = pd.DataFrame(np.asarray(label).flatten())
    print(label.shape)

    time = pd.read_csv("train_order.txt", " ", header=None)
    time = np.asarray(time)

    print(time.shape)
    time = pd.DataFrame(time)
    time = pd.concat(([time] * 6000), axis=1)
    time = pd.DataFrame(np.asarray(time).flatten())
    print(time.shape)

    result = pd.concat((time, acc, o, m, gy, g, l, pressure, label), axis=1)
    print(result.shape)
    # os.chdir(r'data')
    result.to_csv("../dirty.csv", index=False, header=['time',
                                                       'acc_x', 'acc_y', 'acc_z',
                                                       'o_w', 'o_x', 'o_y', 'o_z',
                                                       'm_x', 'm_y', 'm_z',
                                                       'gy_x', 'gy_y', 'gy_z',
                                                       'g_x', 'g_y', 'g_z',
                                                       'l_x', 'l_y', 'l_z',
                                                       'pressure',
                                                       'label'])


def dev():
    os.chdir(r'test/test')
    acc_x = pd.read_csv("Acc_x.txt", ' ', header=None)
    acc_y = pd.read_csv("Acc_y.txt", ' ', header=None)
    acc_z = pd.read_csv("Acc_z.txt", ' ', header=None)
    acc_x = pd.DataFrame(np.asarray(acc_x).flatten())
    acc_y = pd.DataFrame(np.asarray(acc_y).flatten())
    acc_z = pd.DataFrame(np.asarray(acc_z).flatten())

    acc = pd.concat((acc_x, acc_y, acc_z), axis=1)

    print(acc.shape)

    g_x = pd.read_csv("Gra_x.txt", ' ', header=None)
    g_y = pd.read_csv("Gra_y.txt", ' ', header=None)
    g_z = pd.read_csv("Gra_z.txt", ' ', header=None)
    g_x = pd.DataFrame(np.asarray(g_x).flatten())
    g_y = pd.DataFrame(np.asarray(g_y).flatten())
    g_z = pd.DataFrame(np.asarray(g_z).flatten())
    g = pd.concat((g_x, g_y, g_z), axis=1)

    print(g.shape)

    l_x = pd.read_csv("LAcc_x.txt", ' ', header=None)
    l_y = pd.read_csv("LAcc_y.txt", ' ', header=None)
    l_z = pd.read_csv("LAcc_z.txt", ' ', header=None)
    l_x = pd.DataFrame(np.asarray(l_x).flatten())
    l_y = pd.DataFrame(np.asarray(l_y).flatten())
    l_z = pd.DataFrame(np.asarray(l_z).flatten())
    l = pd.concat((l_x, l_y, l_z), axis=1)

    print(l.shape)

    m_x = pd.read_csv("Mag_x.txt", ' ', header=None)
    m_y = pd.read_csv("Mag_y.txt", ' ', header=None)
    m_z = pd.read_csv("Mag_z.txt", ' ', header=None)
    m_x = pd.DataFrame(np.asarray(m_x).flatten())
    m_y = pd.DataFrame(np.asarray(m_y).flatten())
    m_z = pd.DataFrame(np.asarray(m_z).flatten())
    m = pd.concat((m_x, m_y, m_z), axis=1)

    print(m.shape)

    o_w = pd.read_csv("Ori_w.txt", ' ', header=None)
    o_x = pd.read_csv("Ori_x.txt", ' ', header=None)
    o_y = pd.read_csv("Ori_y.txt", ' ', header=None)
    o_z = pd.read_csv("Ori_z.txt", ' ', header=None)
    o_x = pd.DataFrame(np.asarray(o_x).flatten())
    o_y = pd.DataFrame(np.asarray(o_y).flatten())
    o_z = pd.DataFrame(np.asarray(o_z).flatten())
    o_w = pd.DataFrame(np.asarray(o_w).flatten())
    o = pd.concat((o_w, o_x, o_y, o_z), axis=1)

    print(o.shape)

    gy_x = pd.read_csv("Gyr_x.txt", ' ', header=None)
    gy_y = pd.read_csv("Gyr_y.txt", ' ', header=None)
    gy_z = pd.read_csv("Gyr_z.txt", ' ', header=None)
    gy_x = pd.DataFrame(np.asarray(gy_x).flatten())
    gy_y = pd.DataFrame(np.asarray(gy_y).flatten())
    gy_z = pd.DataFrame(np.asarray(gy_z).flatten())
    gy = pd.concat((gy_x, gy_y, gy_z), axis=1)

    print(gy.shape)
    pressure = pd.read_csv("Pressure.txt", ' ', header=None)
    pressure = pd.DataFrame(np.asarray(pressure).flatten())
    print(pressure.shape)

    label = pd.read_csv("Label.txt", ' ', header=None)
    label = pd.DataFrame(np.asarray(label).flatten())
    print(label.shape)

    time = pd.read_csv("test_order.txt", " ", header=None)
    time = np.asarray(time)

    time = time.reshape((time.shape[1], time.shape[0]))

    time = pd.DataFrame(time)
    time = pd.concat(([time] * 6000), axis=1)
    time = pd.DataFrame(np.asarray(time).flatten())
    print(time.shape)

    result = pd.concat((time, acc, o, m, gy, g, l, pressure, label), axis=1)
    print(result.shape)
    # os.chdir(r'data')
    result.to_csv("../dirty.csv", index=False, header=['time',
                                                       'acc_x', 'acc_y', 'acc_z',
                                                       'o_w', 'o_x', 'o_y', 'o_z',
                                                       'm_x', 'm_y', 'm_z',
                                                       'gy_x', 'gy_y', 'gy_z',
                                                       'g_x', 'g_y', 'g_z',
                                                       'l_x', 'l_y', 'l_z',
                                                       'pressure',
                                                       'label'])


if __name__ == '__main__':
    train()
