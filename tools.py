import pandas as pd
import os


def draw():
    os.chdir('data')
    data = pd.read_csv('data_sorted.csv')
    print(data.shape)
    # plt.figure()
    data1 = data[data['label'] == 6]
    # data1 = data1[data['acc_x'] > 40]
    data1 = data1[['acc_x', 'acc_y', 'acc_z']]
    ax = data1.plot(kind='line')
    fig = ax.get_figure()
    fig.savefig("6_acc.png")


if __name__ == '__main__':
    draw()
