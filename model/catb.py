import pandas as pd
import catboost as xgb
from sklearn.metrics import classification_report

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
train_y = y.astype("int")
test_x = pd.read_csv('../test/feature.csv')
test_x = test_x[['gy_z.15', 'pitch.15', 'pitch.3', 'gy_z.3', 'acc_z.15', 'acc_z.13', 'magnetic.2', 'acc_z.3', 'acc_z.2',
                 'pitch.6', 'magnetic.6', 'm_z.15', 'gy_y.13', 'gy_y.8', 'acc_z.6', 'm_z.14', 'm_z.3', 'l_z.11',
                 'gy_z.14',
                 'acc_y.3', 'l_x.3', 'g_z.1', 'magnetic.3', 'l_y.3', 'gy_x.14', 'gy_y.2', 'magnetic.14', 'magnetic.15',
                 'm_x.11',
                 'roll.2', 'acc_z.5', 'roll.6', 'gy_x.4', 'l_y.11', 'gy_z.6', 'l_x.11', 'acc_y.14', 'm_x.15', 'acc_z'
                 ]]

test_y = pd.read_csv("../test/label.csv")
test_y = test_y.astype('int')

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)
print('XGBClassifier------------------------------------------------------------')
xg = xgb.CatBoostClassifier(iterations=2, depth=2, learning_rate=1, loss_function='Logloss',
                            logging_level='Verbose')
xg.fit(train_x, train_y, cat_features=[0,2,5])

print(test_x.shape, test_x.shape)
Prediction_RT = xg.predict(test_x)
print(classification_report(test_y, Prediction_RT, digits=5, ))

# 计算特征值的重要性 全量为87.8%
#
# im = pd.DataFrame(xg.feature_importances_)
# index = pd.DataFrame(train.columns)
#
# im = pd.concat((index, im), axis=1)
# # im.drop(im.columns[1], axis=1, inplace=True)
# # print(index.shape)
# im.columns=['name','im']
# im.sort_values(im.columns[1], inplace=True, ascending=False)
# im.to_csv('importances_all.csv', index=None)

# thresholds = np.sort(xg.feature_importances_)
# thresholds = [0.3]
# for thresh in thresholds:
#     # select features using threshold
#     selection = SelectFromModel(xg, threshold=thresh, prefit=True)
#     select_X_train = selection.transform(train_x)
#     # train model
#     selection_model = xg
#     selection_model.fit(select_X_train, train_y)
#     # eval model
#     select_X_test = selection.transform(test_x)
#     y_pred = selection_model.predict(select_X_test)
#     predictions = [round(value) for value in y_pred]
#     accuracy = accuracy_score(test_y, predictions)
#     print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))
