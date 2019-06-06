import pandas as pd
from tsfresh import extract_features
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters

data = pd.read_csv('data_sorted.csv', nrows=300000)

t = data.groupby('time').mean()
label = t['label']
print(label.shape)
t = pd.Series(label)
print(t)

data.drop('label', axis=1, inplace=True)

print(data.head())
fe = extract_relevant_features(data, t, column_id='time', column_sort='id')
print(fe.head())
print(fe.shape)
fe.to_csv("fe.csv")
