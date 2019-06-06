import keras
import numpy as np
import pandas as pd
from keras import regularizers
from keras.layers import Dense, LSTM, Dropout, Activation, RNN
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

train_x = pd.read_csv('../data/featured.csv')
train_y = pd.read_csv('../data/label.csv')

test_x = pd.read_csv('../test/feature.csv')
test_y = pd.read_csv("../test/label.csv")

for u in train_x.columns:
    train_x[u] = labelencoder.fit_transform(train_x[u])
    if train_x[u].dtype == bool:
        train_x[u] = train_x[u].astype('int')

for u in test_x.columns:
    test_x[u] = labelencoder.fit_transform(test_x[u])
    if test_x[u].dtype == bool:
        test_x[u] = test_x[u].astype('int')
train_x = np.asarray(train_x).reshape((len(train_x), 1, 22 * 16))
test_x = np.asarray(test_x).reshape((len(test_x), 1, 22 * 16))
train_y = keras.utils.to_categorical(train_y, num_classes=9)
test_y = keras.utils.to_categorical(test_y, num_classes=9)

print(train_x.shape)
model = Sequential()
model.add(RNN(32, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(9, activation='softmax'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=32, epochs=100)

pre = model.predict(test_x)
print(pre)
score = model.evaluate(test_x, test_y, batch_size=32)
print("result of LSTM:    ")
print("loss:", score[0], "acc: ", score[1])
