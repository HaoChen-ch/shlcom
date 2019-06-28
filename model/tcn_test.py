import numpy as np
from keras.utils import to_categorical
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def data_generator():
    # input image dimensions
    img_rows, img_cols = 23, 30
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    labelencoder = LabelEncoder()
    x_train = pd.read_csv('../data/feature.csv')
    y_train = pd.read_csv('../data/label.csv')
    x_test = pd.read_csv('../test/feature.csv')
    y_test = pd.read_csv("../test/label.csv")
    for u in x_train.columns:
        x_train[u] = labelencoder.fit_transform(x_train[u])

    for u in x_test.columns:
        x_test[u] = labelencoder.fit_transform(x_test[u])

    x_train = np.asarray(x_train).reshape(-1, img_rows * img_cols, 1)
    x_test = np.asarray(x_test).reshape(-1, img_rows * img_cols, 1)

    num_classes = 9
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    y_train = np.expand_dims(y_train, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    print(y_test)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255

    return (x_train, y_train), (x_test, y_test)


from tcn import compiled_tcn


def run_task():
    (x_train, y_train), (x_test, y_test) = data_generator()
    print("x_train[0:1].shape[1]",x_train[0:1].shape[1])
    model = compiled_tcn(return_sequences=False,
                         num_feat=1,
                         num_classes=9,
                         nb_filters=20,
                         kernel_size=6,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=1,
                         activation='relu',
                         max_len=x_train[0:1].shape[1],
                         use_skip_connections=True)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_test.shape = {y_test.shape}')

    model.summary()

    model.fit(x_train, y_train.squeeze().argmax(axis=1), epochs=50,
              validation_data=(x_test, y_test.squeeze().argmax(axis=1)))


if __name__ == '__main__':

    run_task()
