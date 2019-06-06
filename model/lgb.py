import lightgbm as lgb
import pandas as pd
from sklearn.metrics import classification_report

train = pd.read_csv('../data/feature.csv')
y = pd.read_csv('../data/label.csv')

train_x = train
train_y = y
test_x = pd.read_csv('../test/feature.csv')

test_y = pd.read_csv("../test/label.csv")

# train_x = train_x[['magnetic.14', 'pitch.13', 'acc_z','acc_z.14','m_y.14','acc_z.3','gy_x.14','gy_z.11','gy_y.14',
#                    'magnetic.2','pitch.2','pitch.3','roll.13','acc_z.7','m_y.11','gy_z.14','gy_x.7','acc_y.14','magnetic.3',
#                    'g_x.14','roll.2','acc_z.5','m_z.14','m_y.13','l_x.14','l_x.11','acc_z.9','acc_x.14','magnetic.4',
#                    'pitch.14','m_x.2','m_x.14','roll.14','m_x.3','pitch.9','pitch.4','m_x.4','m_x.11','magnetic.7'
#                    ]]
# test_x = test_x[['magnetic.14', 'pitch.13', 'acc_z','acc_z.14','m_y.14','acc_z.3','gy_x.14','gy_z.11','gy_y.14',
#                    'magnetic.2','pitch.2','pitch.3','roll.13','acc_z.7','m_y.11','gy_z.14','gy_x.7','acc_y.14','magnetic.3',
#                    'g_x.14','roll.2','acc_z.5','m_z.14','m_y.13','l_x.14','l_x.11','acc_z.9','acc_x.14','magnetic.4',
#                    'pitch.14','m_x.2','m_x.14','roll.14','m_x.3','pitch.9','pitch.4','m_x.4','m_x.11','magnetic.7'
#                 ]]
# from imblearn.over_sampling import SMOTE # 导入SMOTE算法模块
# # 处理不平衡数据
# sm = SMOTE(random_state=42)    # 处理过采样的方法
# X, y = sm.fit_sample(X, y.values.ravel())
# train_y = keras.utils.to_categorical(train_y, num_classes=9)
# test_y = keras.utils.to_categorical(test_y, num_classes=9)

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
lg = lgb.LGBMClassifier(
    max_bin=200,
    max_depth=10,
    num_leaves=150,
    learning_rate=0.06
)

fit = lg.fit(train_x, train_y)

Prediction_RT = lg.predict(test_x)
print(classification_report(test_y, Prediction_RT, digits=5))

im = pd.DataFrame(lg.feature_importances_)
index = pd.DataFrame(train.columns)

# im = pd.concat((index, im), axis=1)
# # im.drop(im.columns[1], axis=1, inplace=True)
# # print(index.shape)
# im.columns = ['name', 'im']
# im.sort_values(im.columns[1], inplace=True, ascending=False)
# im.to_csv('importances_all_lg.csv', index=None)
