import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('../data/feature_filter_60.csv')
y = pd.read_csv('../data/label_5.csv')

train_x = train
train_y = y

test_x = pd.read_csv('../test/feature_filter_60.csv')
test_y = pd.read_csv("../test/label_5_all.csv")

print(train_x.shape)
print(test_x.shape)

x = pd.DataFrame(pd.concat((train_x, test_x)).drop(['o_x.26', 'o_y.26', 'o_z.26', 'yaw.26', 'o_w.26'], axis=1)).astype(
    'float')
# pca = PCA(n_components=500)
# reduced_x = pca.fit_transform(x)
reduced_x = x
print(reduced_x.shape)
train_x = reduced_x[:16310]
test_x = reduced_x[16310:22008]
print("reduced   ", train_x.shape)
print("reduced   ", test_x.shape)

lg = lgb.LGBMClassifier(
    max_bin=400,
    max_depth=8,
    num_leaves=1024,
    num_boost_round=1500,
    learning_rate=0.2
)
fit = lg.fit(train_x, train_y)

Prediction_RT = pd.DataFrame(lg.predict(test_x))
predict = pd.concat(([Prediction_RT] * 6000), axis=1)
predict = pd.DataFrame(np.asarray(predict).flatten())

y = pd.read_csv("../test/label_60_ori.csv")

print(classification_report(y, predict, digits=5))
print(classification_report(test_y, Prediction_RT, digits=5))
