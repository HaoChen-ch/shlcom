import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")

train_x = pd.read_csv('../data/feature_unsorted_without_fft_60.csv')
train_y = pd.read_csv('../data/label_unsorted_60.csv')
test_x = pd.read_csv('../test/feature_unsorted_without_fft_60.csv')
test_y = pd.read_csv("../test/label_unsorted_60.csv")

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

eval_result = {}
lg = lgb.LGBMClassifier(
    max_bin=300,
    max_depth=9,
    num_leaves=256,
    num_boost_round=2048,
    boosting_type='goss',
    top_rate=0.06,
    learning_rate=0.1
)

fit = lg.fit(train_x, train_y,
             #   eval_set=[(test_x, test_y)],
             #  early_stopping_rounds=5000,
             callbacks=[lgb.record_evaluation(eval_result)])

Prediction_RT = pd.DataFrame(lg.predict(test_x))

print(classification_report(test_y, Prediction_RT, digits=5))

predict = pd.concat(([Prediction_RT] * 6000), axis=1)
predict = pd.DataFrame(np.asarray(predict).flatten())
print(predict.shape)
y = pd.read_csv("../test/label_unsorted_all.csv")
print(y.shape)
print(classification_report(y, predict, digits=5))  # 使用macro

# print(Prediction_RT.shape)
# Prediction_RT = pd.DataFrame(Prediction_RT.reshape((Prediction_RT.shape[0], 1)))
# reslult = pd.concat((Prediction_RT, test_y), axis=1)
# pd.DataFrame(reslult).to_csv("result_5.csv")

im = pd.DataFrame(lg.feature_importances_)
index = pd.DataFrame(train_x.columns)

im = pd.concat((index, im), axis=1)
im.columns = ['name', 'importance']
im.sort_values(im.columns[1], inplace=True, ascending=False)
im.to_csv('importance_60s.csv', index=None)

# ax = lgb.plot_metric(eval_result)
# plt.savefig("lgb_metric.png")
