import pandas as pd
# from tsfresh.feature_extraction.feature_calculators import count_below_mean, abs_energy, absolute_sum_of_changes, \
#     kurtosis, mean_abs_change, mean_second_derivative_central, sample_entropy, \
#     skewness, variance_larger_than_standard_deviation, percentage_of_reoccurring_values_to_all_values, number_peaks
data = pd.read_csv('test.csv')
#
#
# median = data.groupby('time').median()
# mean = data.groupby('time').mean()
# max = data.groupby('time').max()
# min = data.groupby('time').min()
# var = data.groupby('time').var()
# std = data.groupby('time').std()
#
# absenergy = data.groupby(['time']).apply(
#     lambda group: group.apply(abs_energy)
# )
#
# countbelowmean = data.groupby(['time']).apply(
#     lambda group: group.apply(count_below_mean)
# )
#
# absolutesumofchanges = data.groupby(['time']).apply(
#     lambda group: group.apply(absolute_sum_of_changes)
# )
#
# kurtosi = data.groupby(['time']).apply(
#     lambda group: group.apply(kurtosis)
# )
#
# variancelargerthanstandard_deviation = data.groupby(['time']).apply(
#     lambda group: group.apply(variance_larger_than_standard_deviation)
# )
#
# percentageofreoccurringvaluestoallvalues = data.groupby(['time']).apply(
#     lambda group: group.apply(percentage_of_reoccurring_values_to_all_values)
# )
#
#
# numberpeaks = data.groupby(['time']).apply(
#     lambda group: number_peaks(group, 3)
# )
#
# meanabschange = data.groupby(['time']).apply(
#     lambda group: group.apply(mean_abs_change)
# )

import numpy as np
# x = [9.196213,9.271032,8.998952000000001,8.524157,8.260105000000001,8.53157,8.856849,8.899108,8.518184]
#
# x = pd.Series(x)
# # x= np.asarray(x)
# std4=x.std()*x.std()*x.std()*x.std()
# print(std4)
# x = pd.Series(x)
# p = pd.Series.kurtosis(x)
# print(p)
#
# sum =0
# cnt =0
# for a in x:
#     cnt=cnt+1
#     sum = sum+(a-x.mean())*(a-x.mean())*(a-x.mean())*(a-x.mean())
# q=sum/(std4)
# print(q*(90/(8*7*6))-(3*8*8)/(7*6))


orientation = np.asarray([data['o_w'], data['o_x'], data['o_y'], data['o_z']])
orien = orientation.T
rn0 = np.asarray(1 - 2 * (np.square(orien[:, 2]) + np.square(orien[:, 3])))
rn1 = 2 * (orien[:, 1] * orien[:, 2] - orien[:, 0] * orien[:, 3])
rn2 = 2 * (orien[:, 1] * orien[:, 3] + orien[:, 0] * orien[:, 2])
rn3 = 2 * (orien[:, 1] * orien[:, 2] + orien[:, 0] * orien[:, 3])
rn4 = 1 - 2 * (np.square(orien[:, 1]) + np.square(orien[:, 3]))
rn5 = 2 * (orien[:, 2] * orien[:, 3] - orien[:, 0] * orien[:, 1])
rn6 = 2 * (orien[:, 1] * orien[:, 3] - orien[:, 0] * orien[:, 2])
rn7 = 2 * (orien[:, 2] * orien[:, 3] + orien[:, 0] * orien[:, 1])
rn8 = 1 - 2 * (np.square(orien[:, 1]) + np.square(orien[:, 2]))

o1 = np.asarray([data['o_x'], data['o_y'], data['o_z']])
o_x = pd.DataFrame(rn0 * o1[0] + rn1 * o1[1] + rn2 * o1[2])
o_y = pd.DataFrame(rn3 * o1[0] + rn4 * o1[1] + rn5 * o1[2])
o_z = pd.DataFrame(rn6 * o1[0] + rn7 * o1[1] + rn8 * o1[2])
pitch = pd.DataFrame(np.arctan(rn7 / rn8))
roll = pd.DataFrame(np.arcsin(-rn6))
yaw = pd.DataFrame(np.arctan(rn3 / rn0))
ori = pd.concat((o_x, o_y, o_z, pitch, roll, yaw), axis=1)
print(ori.shape)
