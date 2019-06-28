import lightgbm as lgb
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pylab as plt

train = pd.read_csv('../data/feature.csv')
y = pd.read_csv('../data/label.csv')

train_x = train
train_y = y

test_x = pd.read_csv('../test/feature.csv')
test_y = pd.read_csv("../test/label.csv")
#
# train_x = train_x[
#     ['pressure.9', 'acc_z.4', 'pressure.28', 'gy_x.28', 'magnetic.9', 'gy_z.21', 'magnetic.3', 'm_z.28', 'acc_z.1',
#      'pressure.4', 'pressure.2', 'gy_y.9', 'magnetic.19', 'pitch.27', 'pressure.21', 'l_x.4', 'acc_z.12', 'magnetic.28',
#      'pressure.3',
#      'm_y.28', 'pitch.3', 'acc_x.4', 'gy_x.21', 'o_x.28', 'acc_z.9', 'gy_z.28', 'm_y.21', 'acc_y.4', 'm_x.28',
#      'pressure.19', 'acc_z.28', 'g_x.28', 'acc_z.3', 'acc_x.28', 'l_x.10', 'gy_y.28', 'acc_y.28', 'acc_z.14', 'gy_y.21',
#      'l_x.28', 'gy_z.9', 'acc_y.12', 'acc_z.18', 'm_x.21', 'm_x.4', 'magnetic.2', 'roll.27', 'm_z.3', 'pitch.26',
#      'g_x.19',
#      'acc_z.7', 'm_x.9', 'gy_z.7', 'roll.28', 'm_z.21', 'l_x.21', 'acc_z.8', 'gy_y.3', 'pressure.6', 'pitch.28',
#      'm_y.19', 'pitch.14', 'roll.3'
#      ]]
# test_x = test_x[
#     ['pressure.9', 'acc_z.4', 'pressure.28', 'gy_x.28', 'magnetic.9', 'gy_z.21', 'magnetic.3', 'm_z.28', 'acc_z.1',
#      'pressure.4', 'pressure.2', 'gy_y.9', 'magnetic.19', 'pitch.27', 'pressure.21', 'l_x.4', 'acc_z.12', 'magnetic.28',
#      'pressure.3',
#      'm_y.28', 'pitch.3', 'acc_x.4', 'gy_x.21', 'o_x.28', 'acc_z.9', 'gy_z.28', 'm_y.21', 'acc_y.4', 'm_x.28',
#      'pressure.19', 'acc_z.28', 'g_x.28', 'acc_z.3', 'acc_x.28', 'l_x.10', 'gy_y.28', 'acc_y.28', 'acc_z.14', 'gy_y.21',
#      'l_x.28', 'gy_z.9', 'acc_y.12', 'acc_z.18', 'm_x.21', 'm_x.4', 'magnetic.2', 'roll.27', 'm_z.3', 'pitch.26',
#      'g_x.19',
#      'acc_z.7', 'm_x.9', 'gy_z.7', 'roll.28', 'm_z.21', 'l_x.21', 'acc_z.8', 'gy_y.3', 'pressure.6', 'pitch.28',
#      'm_y.19', 'pitch.14', 'roll.3'
#      ]]
from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
#
# sc.fit(train_x)
#
# train_x = sc.transform(train_x)
# test_x = sc.transform(test_x)

# train_y = to_categorical(train_y,num_classes=9)
# test_y = to_categorical(test_y,num_classes=9)

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
# lg = lgb.LGBMClassifier(
#     max_bin=500,
#     max_depth=15,
#     num_leaves=200,
#     # num_boost_round=1000,
#     learning_rate=0.003
# )
lg = lgb.LGBMClassifier(
    max_bin=250,
    max_depth=10,
    num_leaves=150,
    learning_rate=0.06
)
fit = lg.fit(train_x, train_y)

Prediction_RT = lg.predict(test_x)
print(classification_report(test_y, Prediction_RT, digits=5))
# lg.booster_.save_model("lightgbm63.txt")

# im = pd.DataFrame(lg.feature_importances_)
# index = pd.DataFrame(train.columns)
#
# im = pd.concat((index, im), axis=1)
# # im.drop(im.columns[1], axis=1, inplace=True)
# # print(index.shape)
# im.columns = ['name', 'im']
# im.sort_values(im.columns[1], inplace=True, ascending=False)
# im.to_csv('importances_63_lg.csv', index=None)
# plt.figure(figsize=(12, 6))
# lgb.plot_importance(lg)
# plt.title("Featurertances")
# plt.savefig("lgb_63_feature_importance.png")

# importance = lg.booster_.feature_importance(importance_type='split')
# feature_name = lg.booster_.feature_name()
# feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance})
# feature_importance.sort_values(feature_importance.columns[1], inplace=True, ascending=False)
# feature_importance.to_csv('lgb_63_feature_importance.csv', index=False)
#
# ax = feature_importance.plot(kind='bar', figsize=(30, 10))
# fig = ax.get_figure()
# fig.savefig("t_1all.png")