import pandas as pd
import os


def draw(dir):
    os.chdir(dir)
    data = pd.read_csv('data_sorted.csv')
    print(data.shape)
    # plt.figure()
    data1 = data[data['label'] == 8]
    # data1 = data1[data.values < 40]
    data1 = data1.drop(['time', 'label', 'pressure'], axis=1)
    print(data1.shape)
    ax = data1.plot(kind='line', figsize=(30, 10))
    fig = ax.get_figure()
    fig.savefig("t_8all" + dir + ".png")


def draw1():
    os.chdir('model')
    data = pd.read_csv('lgb_63_feature_importance.csv')
    print(data.shape)
    ax = data.plot(kind='barh', figsize=(30, 10))
    fig = ax.get_figure()
    fig.savefig("t_1all.png")


def draw2():
    data = pd.read_csv('feature_Data_data_filter/max.csv')
    print(data.shape)
    label = pd.read_csv('data/label.csv')
    data_label = pd.concat((data, label), axis=1)
    data_label.drop(['pressure', 'time'], axis=1, inplace=True)
    data_label = data_label[data_label['label'] == 2]
    print(data_label.head)
    ax = data_label.plot(kind='line', figsize=(30, 10))
    fig = ax.get_figure()
    fig.savefig("max_2.png")


def draw3():
    data = pd.read_csv('feature_Data_test/max.csv')
    print(data.shape)
    label = pd.read_csv('test/label.csv')
    data_label = pd.concat((data, label), axis=1)
    data_label.drop(['pressure', 'time'], axis=1, inplace=True)
    data_label = data_label[data_label['label'] == 2]
    print(data_label.head)
    ax = data_label.plot(kind='line', figsize=(30, 10))
    fig = ax.get_figure()
    fig.savefig("max_2_test.png")


def draw4():
    data = pd.read_csv('feature_Data_data_filter/mean.csv')
    print(data.shape)
    label = pd.read_csv('data/label.csv')
    data_label = pd.concat((data, label), axis=1)
    data_label = data_label[["pressure", "label"]]
    data_label = data_label[data_label['label'] == 8]

    data_label = data_label[['pressure']]
    print(data_label.head)
    ax = data_label.plot(kind='line', figsize=(30, 10))
    fig = ax.get_figure()
    fig.savefig("mean_8_train_pressure.png")


if __name__ == '__main__':
    # draw('data')
    #
    # train = pd.read_csv('data/feature.csv')
    # print(train.shape)
    draw4()
