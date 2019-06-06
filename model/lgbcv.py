import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV  # Perforing grid search


train = pd.read_csv('../data/feature.csv')
y = pd.read_csv('../data/label.csv')
# train = train[['gy_z.3', 'acc_z.15', 'pitch.15', 'gy_z.15', 'pitch.3', 'acc_z.13', 'acc_z.3',
#               'magnetic.2', 'acc_z.6', 'acc_y.15', 'm_z.14', 'magnetic.6', 'acc_z.2', 'g_z.4', 'l_y.11', 'gy_y.13',
#               'gy_z.14', 'm_z.3', 'l_y.3', 'gy_y.6', 'g_x.11', 'gy_x.14', 'l_x.11']]
train_x = train
train_y = y
test_x = pd.read_csv('../test/feature.csv')
# test_x = test_x[['gy_z.3', 'acc_z.15', 'pitch.15', 'gy_z.15', 'pitch.3', 'acc_z.13', 'acc_z.3',
#               'magnetic.2', 'acc_z.6', 'acc_y.15', 'm_z.14', 'magnetic.6', 'acc_z.2', 'g_z.4', 'l_y.11', 'gy_y.13',
#               'gy_z.14', 'm_z.3', 'l_y.3', 'gy_y.6', 'g_x.11', 'gy_x.14', 'l_x.11']]

test_y = pd.read_csv("../test/label.csv")
train = lgb.Dataset(train_x, train_y)
valid = lgb.Dataset(test_x, test_y, reference=train)


parameters = {
              'max_depth': [15, 20, 25, 30, 35],
              'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
              'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
              'bagging_freq': [2, 4, 5, 6, 8],
              'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
              'lambda_l2': [0, 10, 15, 35, 40],
              'cat_smooth': [1, 10, 15, 20, 35]
}
gbm = lgb.LGBMClassifier()
# 有了gridsearch我们便不需要fit函数
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='accuracy', cv=3)
gsearch.fit(train_x, train_y)

print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))