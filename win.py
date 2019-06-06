import numpy as np
import pandas as pd
from tsfresh.feature_extraction.feature_calculators import mean

# 整个数据取threshold的窗口，窗口的标记为窗口内部标记数值出现最多次数的值，然后用tsfresh进行特征值提取
# 结果是一个DataFrame

data = pd.read_csv("data_sorted.csv")
# data = pd.read_csv("test.csv")

data = np.asarray(data)
time_begin = data[0, 0]
threshold = 1000  # 窗口 窗口暂时都设为1s 1000ms

rank = np.zeros(9).tolist()  # 创建一个长度为9的数组初始化全是0，下标为运动状态，值为出现的次数
flag = []  # 每组的label

tem_row = []
cnt = 0
# list_row = np.shape((1, 23))
res = []
for row in data:
    li = []
    if row[0] - time_begin < threshold:
        rank[int(row[23])] = rank[int(row[23])] + 1
        i = 0
        while i < 23:
            li.append(row[i])
            i = i + 1
        # tem_row = np.asarray(li)
        tem_row.append(li)
        list_row = np.asarray(tem_row)
        cnt = cnt + 1
    else:
        flag.append(rank.index(max(rank)))  # 求出窗口内标记出现最多的标签，作为最后窗口的标记label
        list_row = pd.DataFrame(list_row)
        # res_row = abs_energy(list_row.iloc[:, 0])
        res_row = mean(list_row)
        res.append(res_row)
        time_begin = row[0]
        cnt = 1
        rank = [0] * 9
        tem_row = []

res = pd.DataFrame(np.asarray(res))
flag = pd.DataFrame(np.asarray(flag))
result = pd.concat((res, flag), axis=1)
print(res.shape, flag.shape, result.shape)
result.to_csv("label.csv", index=False, header=['time', 'acc_x', 'acc_y', 'acc_z',
                                                                             'gy_x', 'gy_y', 'gy_z',
                                                                             'm_x', 'm_y', 'm_z',
                                                                             'o_w', 'o_x', 'o_y', 'o_z',
                                                                             'g_x', 'g_y', 'g_z',
                                                                             'l_x', 'l_y', 'l_z',
                                                                             'press', 'altitude', 'temperature',
                                                                             'label'])
# flag = pd.DataFrame(flag)
