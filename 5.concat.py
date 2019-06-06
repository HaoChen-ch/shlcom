import os
import pandas as pd
import numpy as np

dir = 'feature_Data_test'
files = os.listdir(dir)
os.chdir(dir)
data = pd.DataFrame()
for file in files:
    csv = pd.read_csv(file)
    csv.drop(['time'], inplace=True, axis=1)
    print(csv.shape)
    data = pd.concat((data, csv), axis=1)
print(data.shape)
data.to_csv('../test/feature.csv', index=False)


# dir = 'feature_Data_train'
# files = os.listdir(dir)
# os.chdir(dir)
# order = []
# data = pd.DataFrame()
# for file in files:
#     print(file)
#     order.append(file)
#
# order = pd.DataFrame(np.asarray(order).reshape((16, 1)))
#
# order.to_csv("order.csv", index=True)
