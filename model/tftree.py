import pandas as pd
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
import numpy as np
import warnings

warnings.filterwarnings("ignore")

train_x = pd.read_csv('../data/data_sorted_filter_5.csv')
train_y = pd.read_csv('../data/label_5.csv')

print(train_x.shape, train_y.shape)
train_x['time'] = train_x['time'] - 1

a = np.asarray(['id'])
b = np.asarray(train_x.columns)
a = np.hstack((a, b))


def fun(data):
    id = np.arange(len(data)).reshape((len(data), 1))
    data = pd.DataFrame(np.concatenate((id, data), axis=1))
    return data


train_x = train_x.groupby('time').apply(
    lambda group: fun(group)
)
train_x.columns = a
df = train_x
y = train_y

y = pd.Series(y['label'].values)
# print(y.shape)
print(df.shape)

extraction_settings = ComprehensiveFCParameters()
X = extract_relevant_features(df, y,
                              column_id='time', column_sort='id',
                              default_fc_parameters=extraction_settings
                              )
print(X.head(10))
X.to_csv("extract_features.csv")
# ppl = Pipeline([('fresh', RelevantFeatureAugmenter(column_id='time', column_sort='id')),
#                 ('clf', RandomForestClassifier())])
#
# ppl.set_params(fresh__timeseries_container=df)
# ppl.fit(train_x, train_y)
#
# pred_y = ppl.predict(test_x)
