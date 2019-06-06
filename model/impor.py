import pandas as pd

train = pd.read_csv('../dirty.csv')

im = pd.read_csv('importance.csv')
index = pd.DataFrame(train.columns)

im = pd.concat((index, im), axis=1)
im.drop(im.columns[1], axis=1, inplace=True)
print(index.shape)
im.sort_values(im.columns[1], inplace=True, ascending=False)
im.to_csv('importances.csv', index=None)
