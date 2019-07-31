import pandas as pd
import os
from collections import Counter
import numpy as np


# 考虑这里开始分 窗口 把time drop掉重新设time！
def train():
    os.chdir('data')
    sample = 500
    data = pd.read_csv('raw_data.csv')
    data.sort_values('time', kind='mergesort', inplace=True)
    data.drop(['time'], inplace=True, axis=1)

    time = pd.DataFrame(np.arange(1, (data.shape[0] + 1) / sample))
    time = pd.concat(([time] * sample), axis=1)
    time = pd.DataFrame(np.asarray(time).flatten(), columns=["time"], index=None)
    data = pd.concat((time, pd.DataFrame(data)), axis=1)
    print(data.shape)

    def fun(group):
        label = Counter(group.label).most_common(1)[0][0]
        group = group.loc[lambda s: s.label == label]
        return group

    #
    # # ----------label,顺便改变时间之后就可以设置不同的窗口了
    label = data.groupby(['time']).apply(
        lambda group: Counter(group.label).most_common(1)[0][0]
    )

    label.to_csv('label_5.csv', index=False, header=['label'])
    fin = data.groupby(['time']).apply(
        lambda group: fun(group)
    )
    print(fin.shape)
    fin.to_csv('data_sorted_filter_5.csv', index=False, chunksize=6000000)


def dev():
    sample = 6000
    os.chdir('test')
    data = pd.read_csv('raw_data.csv')
    data.sort_values('time', kind='mergesort', inplace=True)
    data.drop(['time'], inplace=True, axis=1)

    time = pd.DataFrame(np.arange(1, (data.shape[0] + 1) / sample))
    time = pd.concat(([time] * sample), axis=1)
    time = pd.DataFrame(np.asarray(time).flatten(), columns=["time"], index=None)
    data = pd.concat((time, pd.DataFrame(data)), axis=1)
    print(data.shape)

    def fun(group):
        label = Counter(group.label).most_common(1)[0][0]
        group = group.loc[lambda s: s.label == label]
        group = pd.DataFrame(group)
        return group

    # ----------label,顺便改变时间之后就可以设置不同的窗口了
    label = data.groupby(['time']).apply(

        lambda group: Counter(group.label).most_common(1)[0][0]

    )  #

    data = data.groupby(['time']).apply(

        lambda group: fun(group)

    )
    print(label.shape)

    label.to_csv('label_5_all.csv', index=False, header=['label'])
    # data.to_csv('data_sorted_5.csv', index=False, chunksize=6000000)


if __name__ == '__main__':
    # train()
    dev()
