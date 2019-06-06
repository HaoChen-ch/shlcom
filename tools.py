import pandas as pd
import os
import matplotlib.pyplot as plt


def draw():
    os.chdir('data')
    data = pd.read_csv('data_sorted.csv')
    print(data.shape)
    # plt.figure()
    data1 = data[data['label'] == 4]
    # data1 = data1[data.values < 40]
    data1 = data1.drop(['time', 'label', 'pressure'], axis=1)
    print(data1.shape)
    ax = data1.plot(kind='line', figsize=(30, 10))
    fig = ax.get_figure()
    fig.savefig("t_1all.png")


if __name__ == '__main__':
    draw()
