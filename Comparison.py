"""
This script design and test algorithm based on Europe Center Prediction Product
It is for the visibility forecasting project initiated by William Wei (weichen.211@qq.com)
Author: Ben Wang (lbenwong@live.cn)
"""

import pandas as pd
import matplotlib.pylab as plt
import BackTest

from sklearn import linear_model

import warnings


def warn(*args, **kwargs):
    pass
warnings.warn = warn


def data_manipulator(dataset_df):
    # Description: construct feature and clean data
    # Input: pandas data frame
    # Output: pandas data frame
    dataset_df["visia_diff_1"] = dataset_df["visia"].diff(periods=1)
    dataset_df["visia_shift_1"] = dataset_df["visia"].shift(periods=1)
    dataset_df["visia_low"] = dataset_df["visia"] < 3000

    dataset_df["R_1000_diff_1"] = dataset_df["R_1000"].diff(periods=1)
    dataset_df["temp-dew"] = dataset_df["temperature"]-dataset_df["dewtemp"]
    dataset_df["temp-dew_diff_1"] = dataset_df["temp-dew"].diff(periods=1)
    dataset_df["uv_10m_diff_1"] = dataset_df["uv_10m"].diff(periods=1)

    for column in dataset_df:
            dataset_df[column].fillna(dataset_df[column].mean())

    dataset_df = dataset_df.fillna(0)
    return dataset_df[["MSL",
                       "Q_1000", "Q_850", "Q_925",
                       "R_1000", "R_850", "R_925",
                       "T_1000", "T_925", "T_850",
                       "R_1000_diff_1",
                       "temp-dew", "temp-dew_diff_1",
                       "uv_10m", "uv_10m_diff_1",
                       "visia", "visia_diff_1", "visia_shift_1", "visia_low"]]


def linear_regression_func(train, test):
    # Description: The core modelling module
    # input: pandas data frame split into train set and test set
    # output: single test result
    regr = linear_model.LinearRegression()
    regr.fit(train[["R_1000_diff_1", "temp-dew_diff_1", "uv_10m"]], train["visia_diff_1"])
    result = regr.predict(test[["R_1000_diff_1", "temp-dew_diff_1", "uv_10m"]])
    base = test["visia_shift_1"].reset_index()["visia_shift_1"]
    return base[0] + result[0] + result[1] + result[2]
    # return test["visia_shift_1"] < 3000

# Loading and Execution.
EC_DataSet = pd.read_csv("Data/visia.csv")
linear_regression_gt = BackTest.BackTest(
    dataset_df=data_manipulator(EC_DataSet),
    target_col_list=["visia"],
    forecast_func=linear_regression_func)

linear_regression_gt.make_prediction(train_window_int=300, test_window_int=3)
pred = linear_regression_gt.get_prediction_list()
real = linear_regression_gt.get_actual_list()
real = [x.ix[2] for x in real]


# Loading and Execution.
def base_func(train, test):
    base = test["visia_shift_1"].reset_index()["visia_shift_1"]
    return base[0]

base_gt = BackTest.BackTest(
    dataset_df=data_manipulator(EC_DataSet),
    target_col_list=["visia"],
    forecast_func=base_func)

base_gt.make_prediction(train_window_int=300, test_window_int=3)
base_pred = base_gt.get_prediction_list()

plt.figure()
plt.plot(pred, color="blue")
plt.plot(real, color="red")
plt.plot(base_pred, color="green")
plt.title("Blue: Prediction; Red: Real; Green: Shift (Predict 3)")
plt.ylabel("Visibility")
plt.xlabel("Period")
plt.ylim([0, 25000])
plt.show()