import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('../data/featured.csv')
y = pd.read_csv('../data/label.csv')
# train = train[['gy_z.3', 'acc_z.15', 'pitch.15', 'gy_z.15', 'pitch.3', 'acc_z.13', 'acc_z.3',
#               'magnetic.2', 'acc_z.6', 'acc_y.15', 'm_z.14', 'magnetic.6', 'acc_z.2', 'g_z.4', 'l_y.11', 'gy_y.13',
#               'gy_z.14', 'm_z.3', 'l_y.3', 'gy_y.6', 'g_x.11', 'gy_x.14', 'l_x.11']]
train_x = train
train_y = y
test_x = pd.read_csv('../test/feature.csv')
# test_x = test_x[['gy_z.3', 'acc_z.15', 'pitch.15', 'gy_z.15', 'pitch.3', 'acc_z.13', 'acc_z.3',
#               'magnetic.2', 'acc_z.6', 'acc_y.15', 'm_z.14', 'magnetic.6', 'acc_z.2', 'g_z.4', 'l_y.11', 'gy_y.13',
#               'gy_z.14', 'm_z.3', 'l_y.3', 'gy_y.6', 'g_x.11', 'gy_x.14', 'l_x.11']]

test_y = pd.read_csv("../test/label.csv")
# 不要归一化不知道为什么
labelencoder = LabelEncoder()  # 标准化标签，将标签值统一转换成range(标签值个数-1)范围内

for u in train_x.columns:
    train_x[u] = labelencoder.fit_transform(train_x[u])
    # if train_x[u].dtype == bool:
      #  train_x[u] = train_x[u].astype('int')

for u in test_x.columns:
    test_x[u] = labelencoder.fit_transform(test_x[u])
    # if test_x[u].dtype == bool:
        # test_x[u] = test_x[u].astype('int')

# train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=7)
# train_x = pd.concat((train_x, test_x))
# train_y = pd.concat((train_y, test_y))
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

print('RandomForestClassifier------------------------------------------------------------')
RandomForest = RandomForestClassifier()
# parameters = {'n_estimators': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]}
parameters = {'n_estimators': [10]}
clf = GridSearchCV(estimator=RandomForest, param_grid=parameters, cv=3)
clf.fit(train_x, train_y)

Prediction_RT = clf.predict(test_x)

f1 = f1_score(test_y, Prediction_RT, labels=[1, 2, 3, 4, 5, 6, 7, 8], average='micro')
print("f1_score", f1)

print(classification_report(test_y, Prediction_RT, digits=3))
