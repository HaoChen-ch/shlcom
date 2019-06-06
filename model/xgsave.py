import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn2pmml import PMMLPipeline, sklearn2pmml
from xgboost import plot_importance
import matplotlib as plt

train = pd.read_csv('../data/featured.csv')
y = pd.read_csv('../data/label.csv')
train = train[['gy_z.15', 'pitch.15', 'pitch.3', 'gy_z.3', 'acc_z.15', 'acc_z.13', 'magnetic.2', 'acc_z.3', 'acc_z.2',
               'pitch.6', 'magnetic.6', 'm_z.15', 'gy_y.13', 'gy_y.8', 'acc_z.6', 'm_z.14', 'm_z.3', 'l_z.11',
               'gy_z.14',
               'acc_y.3', 'l_x.3', 'g_z.1', 'magnetic.3', 'l_y.3', 'gy_x.14', 'gy_y.2', 'magnetic.14', 'magnetic.15',
               'm_x.11',
               'roll.2', 'acc_z.5', 'roll.6', 'gy_x.4', 'l_y.11', 'gy_z.6', 'l_x.11', 'acc_y.14', 'm_x.15', 'acc_z'
               ]]
train_x = train
train_y = y
test_x = pd.read_csv('../test/feature.csv')
test_x = test_x[['gy_z.15', 'pitch.15', 'pitch.3', 'gy_z.3', 'acc_z.15', 'acc_z.13', 'magnetic.2', 'acc_z.3', 'acc_z.2',
                 'pitch.6', 'magnetic.6', 'm_z.15', 'gy_y.13', 'gy_y.8', 'acc_z.6', 'm_z.14', 'm_z.3', 'l_z.11',
                 'gy_z.14',
                 'acc_y.3', 'l_x.3', 'g_z.1', 'magnetic.3', 'l_y.3', 'gy_x.14', 'gy_y.2', 'magnetic.14', 'magnetic.15',
                 'm_x.11',
                 'roll.2', 'acc_z.5', 'roll.6', 'gy_x.4', 'l_y.11', 'gy_z.6', 'l_x.11', 'acc_y.14', 'm_x.15', 'acc_z'
                 ]]

test_y = pd.read_csv("../test/label.csv")


labelencoder = LabelEncoder()  # 标准化标签，将标签值统一转换成range(标签值个数-1)范围内

# for u in data.columns:
#     # data[u] = labelencoder.fit_transform(data[u])
#     if data[u].dtype == bool:
#         data[u] = data[u].astype('int')
#
# train_x, test_x, train_y, test_y = train_test_split(data, y, test_size=0.2, random_state=7)
#
# data.to_csv('23_feature.csv', index=False)

print(train_y.shape)
print(train_x.shape)

print('XGBClassifier------------------------------------------------------------')
xg = PMMLPipeline([
    ("classifier", xgb.XGBClassifier(max_depth=10,
                                     min_child_weight=1,
                                     gamma=0.1,
                                     subsample=0.8,
                                     colsample_bytree=0.8,
                                     reg_alpha=0.005
                                     ))  # ubicomp2018第二名参数)
])

xg.fit(train_x, train_y)

Prediction_RT = xg.predict(test_x)
print(classification_report(test_y, Prediction_RT, digits=5, ))


sklearn2pmml(xg, "xg2.pmml")

plot_importance(xg,max_num_features=32)
print(xg.feature_importances_)
plt.show()
plt.savefig('importance', dpi=600)
