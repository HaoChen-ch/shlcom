import pandas as pd
import os
from collections import Counter
import numpy as np

# 考虑这里开始分 窗口 把time drop掉重新设time！
os.chdir('data')
data = pd.read_csv('raw_data.csv')
data.sort_values('time', kind='mergesort', inplace=True)


# os.chdir('test')

def fun(group):
    label = Counter(group.label).most_common(1)[0][0]
    group = group[group['label'] == label]
    return group


# ----------label,顺便改变时间之后就可以设置不同的窗口了
# label = data.groupby(['time']).apply(
#
#     lambda group: Counter(group.label).most_common(1)[0][0]
#
# )

data = data.groupby(['time']).apply(

    lambda group: fun(group)

)

print(data.shape)

# label.to_csv('label.csv', index=False, header=['label'])
data.to_csv('data_sorted_filter.csv', chunksize=60000, index=False)
