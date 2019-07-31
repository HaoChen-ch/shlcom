import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report

train = pd.read_csv('../data/feature_filter.csv')
y = pd.read_csv('../data/label.csv')
# train = train[['gy_z.15', 'pitch.15', 'pitch.3', 'gy_z.3', 'acc_z.15', 'acc_z.13', 'magnetic.2', 'acc_z.3', 'acc_z.2',
#                'pitch.6', 'magnetic.6', 'm_z.15', 'gy_y.13', 'gy_y.8', 'acc_z.6', 'm_z.14', 'm_z.3', 'l_z.11',
#                'gy_z.14',
#                'acc_y.3', 'l_x.3', 'g_z.1', 'magnetic.3', 'l_y.3', 'gy_x.14', 'gy_y.2', 'magnetic.14', 'magnetic.15',
#                'm_x.11',
#                'roll.2', 'acc_z.5', 'roll.6', 'gy_x.4', 'l_y.11', 'gy_z.6', 'l_x.11', 'acc_y.14', 'm_x.15', 'acc_z'
#                ]]
# train = train[['gy_z.15', 'pitch.15', 'pitch.3', 'gy_z.3', 'acc_z.15', 'acc_z.13', 'magnetic.2', 'acc_z.3', 'acc_z.2',
#                'pitch.6', 'magnetic.6', 'm_z.15', 'gy_y.13', 'acc_z.6', 'm_z.14', 'm_z.3', 'l_z.11',
#                'gy_z.14',
#                'acc_y.3', 'l_x.3', 'g_z.1', 'magnetic.3', 'l_y.3'
#                ]]
train_x = train
train_y = y
test_x = pd.read_csv('../test/feature.csv')
# test_x = test_x[['gy_z.15', 'pitch.15', 'pitch.3', 'gy_z.3', 'acc_z.15', 'acc_z.13', 'magnetic.2', 'acc_z.3', 'acc_z.2',
#                  'pitch.6', 'magnetic.6', 'm_z.15', 'gy_y.13', 'acc_z.6', 'm_z.14', 'm_z.3', 'l_z.11',
#                  'gy_z.14',
#                  'acc_y.3', 'l_x.3', 'g_z.1', 'magnetic.3', 'l_y.3'
#                  ]]

test_y = pd.read_csv("../test/label.csv")

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
print('XGBClassifier------------------------------------------------------------')
# xg = xgb.XGBClassifier(max_depth=10,
#                        min_child_weight=1,
#                        gamma=0.1,
#                        subsample=0.8,
#                        colsample_bytree=0.8,
#                         # number_of_trees=5000,
#                        reg_alpha=0.0005
#                        )  # ubicomp2018第二名参数

xg = xgb.XGBClassifier(max_depth=10,
                       min_child_weight=1,
                       gamma=0.1,
                       subsample=0.65,
                       colsample_bytree=0.8,
                       # number_of_trees=2500,
                       reg_alpha=0.15
                       )  # ubicomp2018第二名参数
# xg = xgb.XGBClassifier(max_depth=10,
#                        min_child_weight=1,
#                        gamma=0,
#                        subsample=0.65,
#                        colsample_bytree=0.75,
#                        number_of_trees=5000,
#                        reg_alpha=0.0005  #0.00001
#                        )  # ubicomp2018第二名参数
xg.fit(train_x.astype("float64"), train_y.astype("int"))

print(test_x.shape, test_x.shape)
Prediction_RT = xg.predict(test_x.astype("float64"))
print(classification_report(test_y, Prediction_RT, digits=5, ))


# 计算特征值的重要性 全量为87.8%
#
# im = pd.DataFrame(xg.feature_importances_)
# index = pd.DataFrame(train.columns)
#
# im = pd.concat((index, im), axis=1)
# # im.drop(im.columns[1], axis=1, inplace=True)
# # print(index.shape)
# im.columns = ['name', 'importances']
# im.sort_values(im.columns[1], inplace=True, ascending=False)
# im.to_csv('importances_60s.csv', index=None)
