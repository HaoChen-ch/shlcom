import pandas as pd
import numpy as np
import os
from tsfresh.feature_extraction.feature_calculators import *

# data = pd.read_csv("data/data_sorted_filter_5.csv")
# os.chdir('feature_Data_train_5')
data = pd.read_csv("test/data_sorted_5.csv")
os.chdir('feature_Data_test_5')
data.drop(['label'], axis=1, inplace=True)
print(data.shape)
data = data.drop(index=[0])
data = data[0:len(data)].astype("float64")
print(data.dtypes)


# # ---------------------------------fft_coefficient------------------------------------------

def fun_fft_coefficient(data, param):
    re = data.apply(lambda x: fft_coefficient(x, [{'coeff': 2, 'attr': param}]))
    re = re.apply(lambda x: list(*zip(x))[0][1])
    return re


# result = fun_fft_coefficient(data, "real")
fft_coefficient_real = data.groupby(['time']).apply(
    lambda group: fun_fft_coefficient(group, "real")
)
fft_coefficient_real.drop(['time'], axis=1, inplace=True)
print(fft_coefficient_real.shape)
fft_coefficient_real.to_csv('fft_coefficient_real.csv', index_label='time')

fft_coefficient_imag = data.groupby(['time']).apply(
    lambda group: fun_fft_coefficient(group, "imag")
)
fft_coefficient_imag.drop(['time'], axis=1, inplace=True)
print(fft_coefficient_imag.shape)
fft_coefficient_imag.to_csv('fft_coefficient_imag.csv', index_label='time')

fft_coefficient_abs = data.groupby(['time']).apply(
    lambda group: fun_fft_coefficient(group, "abs")
)
fft_coefficient_abs.drop(['time'], axis=1, inplace=True)
print(fft_coefficient_abs.shape)
fft_coefficient_abs.to_csv('fft_coefficient_abs.csv', index_label='time')

fft_coefficient_angle = data.groupby(['time']).apply(
    lambda group: fun_fft_coefficient(group, "angle")
)
fft_coefficient_angle.drop(['time'], axis=1, inplace=True)
print(fft_coefficient_angle.shape)
fft_coefficient_angle.to_csv('fft_coefficient_angle.csv', index_label='time')

print("fft_coefficient")


# ---------------------------------time_reversal_asymmetry_statistic------------------------------------------
def fun_time_reversal_asymmetry_statistic(data):
    re = data.apply(lambda x: time_reversal_asymmetry_statistic(x, 200))
    # re = re.apply(lambda x: list(*zip(x))[0][1])
    return re


# print(fun_time_reversal_asymmetry_statistic(data))
time_reversal_asymmetry_statistic = data.groupby(['time']).apply(
    lambda group: fun_time_reversal_asymmetry_statistic(group)
)
time_reversal_asymmetry_statistic.drop(['time'], axis=1, inplace=True)
print(time_reversal_asymmetry_statistic.shape)
time_reversal_asymmetry_statistic.to_csv('time_reversal_asymmetry_statistic.csv', index_label='time')
print("time_reversal_asymmetry_statistic")


# ---------------------------------cid_ce------------------------------------------
def fun_cid_ce(data):
    re = data.apply(lambda x: cid_ce(x, True))
    # re = re.apply(lambda x: list(*zip(x))[0][1])
    return re


# print(fun_cid_ce(data))
cid_ce = data.groupby(['time']).apply(
    lambda group: fun_cid_ce(group)
)
cid_ce.drop(['time'], axis=1, inplace=True)
print(cid_ce.shape)
cid_ce.to_csv('cid_ce.csv', index_label='time')
print("cid_ce")


# ---------------------------------autocorrelation------------------------------------------
def fun_autocorrelation(data):
    re = data.apply(lambda x: autocorrelation(x, 200))
    # re = re.apply(lambda x: list(*zip(x))[0][1])
    return re


# print(fun_autocorrelation(data))
autocorrelation = data.groupby(['time']).apply(
    lambda group: fun_autocorrelation(group)
)
autocorrelation.drop(['time'], axis=1, inplace=True)
print(autocorrelation.shape)
autocorrelation.to_csv('autocorrelation.csv', index_label='time')
print("autocorrelation")


# ---------------------------------ar_coefficient--------shijianchang ----------------------------------
# def fun_ar_coefficient(data):
#     re = data.apply(lambda x: ar_coefficient(x, [{"k": 200, "coeff": 2}]))
#     re = re.apply(lambda x: list(*zip(x))[0][1])
#     return re
#
#
# # print(fun_ar_coefficient(data))
# ar_coefficient = data.groupby(['time']).apply(
#     lambda group: fun_ar_coefficient(group)
# )
# ar_coefficient.drop(['time'], axis=1, inplace=True)
# print(ar_coefficient.shape)
# ar_coefficient.to_csv('ar_coefficient.csv', index_label='time')

# ---------------------------------ratio_beyond_r_sigma------------------------------------------
def fun_ratio_beyond_r_sigma(data):
    re = data.apply(lambda x: ratio_beyond_r_sigma(x, 1))
    return re


# print(fun_ratio_beyond_r_sigma(data))
ratio_beyond_r_sigma = data.groupby(['time']).apply(
    lambda group: fun_ratio_beyond_r_sigma(group)
)
ratio_beyond_r_sigma.drop(['time'], axis=1, inplace=True)
print(ratio_beyond_r_sigma.shape)
ratio_beyond_r_sigma.to_csv('ratio_beyond_r_sigma.csv', index_label='time')
print("ratio_beyond_r_sigma")


# ---------------------------------spkt_welch_density------------------------------------------
def fun_spkt_welch_density(data):
    param = [{"coeff": 1}]
    re = data.apply(lambda x: spkt_welch_density(x=x, param=param))
    re = re.apply(lambda x: list(*zip(x))[0][1])
    return re


spkt_welch_density = data.groupby(['time']).apply(
    lambda group: fun_spkt_welch_density(group)
)
spkt_welch_density.drop(['time'], axis=1, inplace=True)
print(spkt_welch_density.shape)
spkt_welch_density.to_csv('spkt_welch_density.csv', index_label='time')
print("spkt_welch_density")
# ---------------------------------partial_autocorrelation--------shijianchang ----------------------------------
# def fun_partial_autocorrelation(x):
#     re = x.apply(lambda x: partial_autocorrelation(x, [{"lag": 1}]))
#     re = re.apply(lambda x: list(*zip(x))[0][1])
#     return re
#
#
# print(fun_partial_autocorrelation(data))
# partial_autocorrelation = data.groupby(['time']).apply(
#     lambda group: fun_partial_autocorrelation(group)
# )
# partial_autocorrelation.drop(['time'], axis=1, inplace=True)
# print(partial_autocorrelation.shape)
# partial_autocorrelation.to_csv('partial_autocorrelation.csv', index_label='time')


# ---------------------------------number_cwt_peaks----------------shijianchang --------------------------
# def fun_number_cwt_peaks(x):
#     re = x.apply(lambda x: number_cwt_peaks(x, 5))
#     return re
#
#
# number_cwt_peaks = data.groupby(['time']).apply(
#     lambda group: fun_number_cwt_peaks(group)
# )
# number_cwt_peaks.drop(['time'], axis=1, inplace=True)
# print(number_cwt_peaks.shape)
# number_cwt_peaks.to_csv('number_cwt_peaks.csv', index_label='time')

# # ---------------------------------mean_second_derivative_central------------------------------------------
mean_second_derivative_central = data.groupby(['time']).apply(
    lambda group: group.apply(mean_second_derivative_central)
)
mean_second_derivative_central.drop(['time'], axis=1, inplace=True)
print(mean_second_derivative_central.shape)
mean_second_derivative_central.to_csv('mean_second_derivative_central.csv', index_label='time')
print("mean_second_derivative_central")


# # ---------------------------------index_mass_quantile------------------------------------------
def fun_index_mass_quantile(x):
    re = x.apply(lambda x: index_mass_quantile(x, [{"q": 0.65}]))
    re = re.apply(lambda x: list(*zip(x))[0][1])
    return re


# print(fun(data))

index_mass_quantile = data.groupby(['time']).apply(
    lambda group: fun_index_mass_quantile(group)
)

index_mass_quantile.drop(['time'], axis=1, inplace=True)
print(index_mass_quantile.shape)
index_mass_quantile.to_csv('index_mass_quantile.csv', index_label='time')
print("index_mass_quantile")


# # # ---------------------------------fft_aggregated------------------------------------------

#
def fun(x, f):
    re = x.apply(lambda x: fft_aggregated(x, [{"aggtype": f}]))
    re = re.apply(lambda x: list(*zip(x))[0][1])
    return re


# --------------------------------------------------------------
fft_aggregated_centroid = data.groupby(['time']).apply(
    lambda group: fun(group, "centroid")
)

fft_aggregated_centroid.drop(['time'], axis=1, inplace=True)
print(fft_aggregated_centroid.shape)
fft_aggregated_centroid.to_csv('fft_aggregated_centroid.csv', index_label='time')
print("fft_aggregated_centroid")
#  ------------------------------------------------------------
fft_aggregated_skew = data.groupby(['time']).apply(
    lambda group: fun(group, "skew")
)

fft_aggregated_skew.drop(['time'], axis=1, inplace=True)
print(fft_aggregated_skew.shape)
fft_aggregated_skew.to_csv('fft_aggregated_skew.csv', index_label='time')
print("fft_aggregated_skew")
#  -----------------------------------------------------------------------
fft_aggregated_kurtosis = data.groupby(['time']).apply(
    lambda group: fun(group, "kurtosis")
)

fft_aggregated_kurtosis.drop(['time'], axis=1, inplace=True)
print(fft_aggregated_kurtosis.shape)
fft_aggregated_kurtosis.to_csv('fft_aggregated_kurtosis.csv', index_label='time')
print("fft_aggregated_kurtosis")


# # # ---------------------------------fft_aggregated-variance------------------------------------------
#

def fun_variance(x):
    re = x.apply(lambda x: fft_aggregated(x, [{"aggtype": "variance"}]))
    re = re.apply(lambda x: list(*zip(x))[0][1])
    return re


fft_aggregated_variance = data.groupby(['time']).apply(
    lambda group: fun_variance(group)
)

fft_aggregated_variance.drop(['time'], axis=1, inplace=True)
print(fft_aggregated_variance.shape)
fft_aggregated_variance.to_csv('fft_aggregated_variance.csv', index_label='time')
print('fft_aggregated_variance')
# # ---------------------------------approximate_entropy----meiyousuan --------------------------------------
# approximate_entropy = data.groupby(['time']).apply(
#     lambda group: group.apply(lambda x: approximate_entropy(x, m=5, r=1))
# )
# approximate_entropy.drop(['time'], axis=1, inplace=True)
# print(approximate_entropy.shape)
#
# approximate_entropy.to_csv('approximate_entropy.csv', index_label='time')

# # ---------------------------------abs_energy------------------------------------------
abs_energy = data.groupby(['time']).apply(
    lambda group: group.apply(abs_energy)
)
abs_energy.drop(['time'], axis=1, inplace=True)
print(abs_energy.shape)
abs_energy.to_csv('abs_energy.csv', index_label='time')
print('abs_energy')
#
# # ----------------------------------------numpy的统计函数-------------------------------------------------
#
median = data.groupby('time').median()
mean = data.groupby('time').mean()
max = data.groupby('time').max()
min = data.groupby('time').min()
var = data.groupby('time').var()
std = data.groupby('time').std()
median.to_csv('median.csv', index_label='time')
mean.to_csv('mean.csv', index_label='time')
max.to_csv('max.csv', index_label='time')
min.to_csv('min.csv', index_label='time')
var.to_csv('var.csv', index_label='time')
std.to_csv('std.csv', index_label='time')
print('numpy')
# # # ---------------------------------------------------------------------------------------------------
count_below_mean = data.groupby(['time']).apply(
    lambda group: group.apply(count_below_mean)
)
count_below_mean.drop(['time'], axis=1, inplace=True)
print(count_below_mean.shape)
count_below_mean.to_csv('count_below_mean.csv', index_label='time')
print('count_below_mean')
# #
# # # --------------------------------------------------------------------------------------------------------------
# # #
absolute_sum_of_changes = data.groupby(['time']).apply(
    lambda group: group.apply(absolute_sum_of_changes)
)
absolute_sum_of_changes.drop(['time'], axis=1, inplace=True)

print(absolute_sum_of_changes.shape)
absolute_sum_of_changes.to_csv('absolute_sum_of_changes.csv', index_label='time')
print('absolute_sum_of_changes')

# --------------------------------------------------------------------------------------------------------------

kurtosis = data.groupby(['time']).apply(
    lambda group: group.apply(kurtosis)
)
kurtosis.drop(['time'], axis=1, inplace=True)

print(kurtosis.shape)
kurtosis.to_csv('kurtosis.csv', index_label='time')
print('kurtosis')

# # #
# # # # --------------------------------------------------------------------------------------------------------------
# # #
mean_abs_change = data.groupby(['time']).apply(
    lambda group: group.apply(mean_abs_change)
)
mean_abs_change.drop(['time'], axis=1, inplace=True)

print(mean_abs_change.shape)
mean_abs_change.to_csv('mean_abs_change.csv', index_label='time')
print('mean_abs_change')
# #
# # # # --------------------有问题----算的太慢？-------------------------------------------------------
# # sample_entropy = data.groupby(['time']).apply(
# #     lambda group: group.apply(sample_entropy)
# # )
# # sample_entropy.drop(['time'], axis=1, inplace=True)
# #
# # print(sample_entropy.shape)
# # sample_entropy.to_csv('sample_entropy.csv', index_label='time')
#
# # # # -------------------------------------------------------------------------------
skewness = data.groupby(['time']).apply(
    lambda group: group.apply(skewness)
)
skewness.drop(['time'], axis=1, inplace=True)

print(skewness.shape)
skewness.to_csv('skewness.csv', index_label='time')
print('skewness')
# # #
# # # # -------------------------------------------------------------------------------
variance_larger_than_standard_deviation = data.groupby(['time']).apply(
    lambda group: group.apply(variance_larger_than_standard_deviation)
)
variance_larger_than_standard_deviation.drop(['time'], axis=1, inplace=True)

print(variance_larger_than_standard_deviation.shape)
variance_larger_than_standard_deviation.to_csv('variance_larger_than_standard_deviation.csv', index_label='time')
print('variance_larger_than_standard_deviation')
# # # #
# # # # -------------------------------------------------------------------------------
percentage_of_reoccurring_values_to_all_values = data.groupby(['time']).apply(
    lambda group: group.apply(percentage_of_reoccurring_values_to_all_values)
)
percentage_of_reoccurring_values_to_all_values.drop(['time'], axis=1, inplace=True)

percentage_of_reoccurring_values_to_all_values.to_csv('percentage_of_reoccurring_values_to_all_values.csv',
                                                      index_label='time')
print(percentage_of_reoccurring_values_to_all_values.shape)
print('percentage_of_reoccurring_values_to_all_values')
# #
# # #-------------------------------------------------------------------------------
number_peaks = data.groupby(['time']).apply(
    lambda group: number_peaks(group, 3)
)
number_peaks.drop(['time'], axis=1, inplace=True)

print(number_peaks.shape)
number_peaks.to_csv('number_peaks.csv', index_label='time')
print('number_peaks')
