import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, RNN, GRU, Reshape, BatchNormalization
from keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import os

# os.system("srun --gres=gpu:V100:4")

labelencoder = LabelEncoder()

train = pd.read_csv('../data/feature.csv')
y = pd.read_csv('../data/label.csv')

test_x = pd.read_csv('../test/feature.csv')
test_y = pd.read_csv("../test/label.csv")

for u in train.columns:
    train[u] = labelencoder.fit_transform(train[u])
    if train[u].dtype == bool:
        train[u] = train[u].astype('int')

for u in test_x.columns:
    test_x[u] = labelencoder.fit_transform(test_x[u])
    if test_x[u].dtype == bool:
        test_x[u] = test_x[u].astype('int')


print(train.shape, y.shape)

Train_X = train
Test_X = test_x
Train_Y = y
Test_Y_ori = test_y
nb_features = 23
nb_class = 30
Train_X = Train_X.append(Train_X)
Train_Y = Train_Y.append(Train_Y)

Train_X = np.asarray(Train_X).reshape((len(Train_X), 1, nb_class, nb_features))
Test_X = np.asarray(Test_X).reshape((len(Test_X), 1, nb_class, nb_features))

print(Train_X.shape)

Train_Y = keras.utils.to_categorical(Train_Y, num_classes=9)
Test_Y_ori = keras.utils.to_categorical(Test_Y_ori, num_classes=9)

# Model and Compile
acti = 'relu'
model = Sequential()
model.add(Conv2D(32, 3, strides=(2, 2), kernel_initializer='he_normal', padding='same', activation='relu',
                 input_shape=(1, nb_class, nb_features)))
model.add(Conv2D(32, 3, strides=(2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D(strides=(1, 2), padding='same'))
# model.add(Dropout(.5))
# model.add(Dropout(.5))
# model.add(BatchNormalization())
model.add(Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D(strides=(1, 2), padding='same'))
# model.add(Dropout(.5))
# model.add(Dropout(.5))
# model.add(BatchNormalization())
model.add(Conv2D(128, 3, strides=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(128, 3, strides=(2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D(strides=(1, 2), padding='same'))
# model.add(Dropout(.5))
# model.add(BatchNormalization())
model.add(Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D(strides=(1, 2), padding='same'))
model.add(Dropout(.3))
# model.add(BatchNormalization())
model.add(Conv2D(32, 3, strides=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(32, 3, strides=(2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D(strides=(1, 2), padding='same'))
model.add(Dropout(.3))
#
# model.add(BatchNormalization())
# model.add(Conv2D(256, 3, strides=(2, 2), padding='same', activation='relu'))
# model.add(Conv2D(256, 3, strides=(2, 2), padding='same', activation='relu'))
# model.add(MaxPooling2D(strides=(2, 2), padding='same'))
# model.add(Dropout(.5))
#
# model.add(BatchNormalization())
# model.add(Conv2D(512, 3, strides=(2, 2), padding='same', activation='relu'))
# model.add(Conv2D(512, 3, strides=(2, 2), padding='same', activation='relu'))
# model.add(MaxPooling2D(strides=(2, 2), padding='same'))
# model.add(Dropout(.5))
#
# model.add(BatchNormalization())
model.add(Flatten())
# A = model.output_shape
# model.add(Dense(int(A[1] * 1 / 4.), activation='relu'))
# model.add(Dropout(.5))
# print(model.output_shape)

# # CNN后面接上RNN
# model.add(Reshape((1, 512)))
# print(model.output_shape)
# model.add(GRU(64, recurrent_dropout=0.1, return_sequences=True))
# model.add(GRU(32, recurrent_dropout=0.1))# 128 64 70%
# model.add(BatchNormalization())
# model.add(Dense(64, activation=acti))
# model.add(Dense(32, activation=acti))
model.add(Dense(16, activation=acti))
# model.add(Dropout(0.3))
model.add(Dense(9, activation='softmax'))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Ensemble configuration
model.fit(Train_X, Train_Y, batch_size=64, epochs=100, validation_data=(Test_X, Test_Y_ori))

pre = model.predict(Test_X)

score = model.evaluate(Test_X, Test_Y_ori, batch_size=64)
print("loss:", score[0], "acc: ", score[1])
