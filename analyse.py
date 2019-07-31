import pandas as pd
import os

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
from sklearn.metrics import classification_report
import numpy as np

predict = pd.read_csv("model/result.csv")
y = pd.read_csv("test/label_60_all.csv", header=None)
print(y.shape)
predict = pd.concat(([predict['0']] * 6000), axis=1)
predict = pd.DataFrame(np.asarray(predict).flatten())
print(classification_report(y, predict, digits=5))
