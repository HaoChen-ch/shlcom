import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report
from collections import Counter
# data = pd.read_csv("result_5.csv")
# print("data.shape : ", data.shape)
#
# l5p6 = data.loc[lambda s: s.label == 5].loc[lambda s: s.predict == 6]
# print("l5p6.shape : ", l5p6.shape)
#
# l6p5 = data.loc[lambda s: s.label == 6].loc[lambda s: s.predict == 5]
# print("l6p5.shape : ", l6p5.shape)
#
# l7p8 = data.loc[lambda s: s.label == 7].loc[lambda s: s.predict == 8]
# print("l7p8.shape : ", l7p8.shape)
#
# l8p7 = data.loc[lambda s: s.label == 8].loc[lambda s: s.predict == 7]
# print("l8p7.shape : ", l8p7.shape)
#
# p1 = data.loc[lambda s: s.predict == 1]
# print("p1.shape : ", p1.shape)
#
# l1 = data.loc[lambda s: s.label == 1]
# print("l1.shape : ", l1.shape)

# from sklearn.metrics import classification_report
# import numpy as np
# from hmmlearn import hmm
# model = hmm.GaussianHMM (n_components=9)
#
# predict = pd.read_csv("model/result.csv")
# test_y = pd.read_csv("test/label.csv").astype(int)
# # predict = pd.concat(([predict['0']] * 6000), axis=1)
# predict = pd.DataFrame(predict['0']).astype(int)
# print(classification_report(test_y, predict, digits=5))
# print(predict.shape,test_y.shape)
# model.fit(predict)
# yhat = model.predict(predict)+1
# print(yhat.shape)
#
# print(classification_report(test_y, yhat, digits=5))

predict = pd.read_csv("model/prediction_windows5.csv")
predict.drop(['time'], inplace=True, axis=1)
time = pd.DataFrame(np.arange(0, 5698))
time = pd.concat(([time] * 12), axis=1)
time = pd.DataFrame(np.asarray(time).flatten(), columns=["time"], index=None)
data = pd.concat((time, pd.DataFrame(predict)), axis=1)
data = data.groupby(['time']).apply(
    lambda x: max(x['label'])
)
# data = data.groupby(['time']).apply(
#     lambda x: Counter(x.label).most_common(1)[0][0]
# )

data = pd.concat(([predict] * 500), axis=1)
data = pd.DataFrame(np.asarray(data).flatten())
y = pd.read_csv("test/label_60_all.csv", header=None)
print(data.shape, y.shape)
print(classification_report(y, data, digits=5))

# print(predict.shape)
# print(time.shape)
# print(y.shape)
