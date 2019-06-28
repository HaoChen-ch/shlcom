import pandas as pd
import os
from collections import Counter
import numpy as np

# 考虑这里开始分 窗口 把time drop掉重新设time！
def train():
    os.chdir('data')
    data = pd.read_csv('raw_data.csv')
    data.sort_values('time', kind='mergesort', inplace=True)


# os.chdir('test')

    def fun(group):
        label = Counter(group.label).most_common(1)[0][0]
        # group = group[group['label'] == label]
        group = group.loc[lambda s: s.label == label]
        group = pd.DataFrame(group)
        # print(group.shape)
        # group.to_csv('data_sorted_filter.csv', mode='a', index=False, header=0)
        return group

    #
    # # ----------label,顺便改变时间之后就可以设置不同的窗口了
    label = data.groupby(['time']).apply(
        lambda group: Counter(group.label).most_common(1)[0][0]
    )

    label.to_csv('label.csv', index=False, header=['label'])
    fin = data.groupby(['time']).apply(
        lambda group: fun(group)
    )
    print(fin.shape)
    fin.to_csv('data_sorted_filter.csv', index=False, chunksize=6000000)



def dev():
    os.chdir('test')
    data = pd.read_csv('raw_data.csv')
    data.sort_values('time', kind='mergesort', inplace=True)

    # def fun(group):
    #     label = Counter(group.label).most_common(1)[0][0]
    #     group = group[group['label'] == label]
    #     return group

    # ----------label,顺便改变时间之后就可以设置不同的窗口了
    label = data.groupby(['time']).apply(

        lambda group: Counter(group.label).most_common(1)[0][0]

    )  #
    data.to_csv('data_sorted.csv', index=False)

    # data = data.groupby(['time']).apply(
    #
    #     lambda group: fun(group)
    #
    # )

    print(label.shape)

    # label.to_csv('label.csv', index=False, header=['label'])
    # data.to_csv('data_sorted.csv', index=False)


if __name__ == '__main__':
    print("kaishi ")
    train()
