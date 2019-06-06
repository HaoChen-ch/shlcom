import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, RNN, GRU, Reshape, BatchNormalization
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

# Train_X = np.asarray(Train_X).reshape((len(Train_X), 690))
# Test_X = np.asarray(Test_X).reshape((len(Test_X), 690))

print(Train_X.shape)

Train_Y = keras.utils.to_categorical(Train_Y, num_classes=9)
Test_Y_ori = keras.utils.to_categorical(Test_Y_ori, num_classes=9)

# Model and Compile
acti = 'tanh'
model = Sequential()

model.add(Dense(32, activation=acti, input_dim=690))
model.add(Dense(32, activation=acti))
model.add(Dense(32, activation=acti))
model.add(Dense(16, activation=acti))
model.add(Dense(16, activation=acti))
model.add(Dense(9, activation='softmax'))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Ensemble configuration
model.fit(Train_X, Train_Y, batch_size=64, epochs=100, validation_data=(Test_X, Test_Y_ori))

pre = model.predict(Test_X)

score = model.evaluate(Test_X, Test_Y_ori, batch_size=64)
print("loss:", score[0], "acc: ", score[1])
