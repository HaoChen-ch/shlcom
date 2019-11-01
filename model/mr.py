# import lightgbm as lgb
import pandas as pd
import pymrmr  # python 使用3.6版本的
import warnings

warnings.filterwarnings("ignore")

train = pd.read_csv('../data/feature_filter_60.csv', nrows=50)
y = pd.read_csv('../data/label_5.csv', nrows=50)

train_x = train
train_y = y

test_x = pd.read_csv('../test/feature_filter_60.csv', nrows=50)
test_y = pd.read_csv("../test/label_5_all.csv", nrows=50)
print(train_x.shape, test_x.shape)
data = pd.DataFrame(
    pd.concat((train_x, test_x)).drop(['o_x.26', 'o_y.26', 'o_z.26', 'yaw.26', 'o_w.26'], axis=1)).astype('int32')

# data = data[0:len(data)].astype("int32")  # .astype(str)
label = pd.concat((train_y, test_y), axis=0)
print(data.shape, label.shape)
df = pd.concat((label, data), axis=1)
print(df.shape)

res = pymrmr.mRMR(data, 'MIQ', 500)

print(res.shape)
