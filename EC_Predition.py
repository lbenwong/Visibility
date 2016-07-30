"""
This script design and test algorithm based on Europe Center Prediction Product
It is for the visibility forecasting project initiated by William Wei (weichen.211@qq.com)
Author: Ben Wang (lbenwong@live.cn)
"""

import pandas as pd
import matplotlib.pyplot as plt
import BackTest

from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

import warnings


def warn(*args, **kwargs):
    pass
warnings.warn = warn


def data_manipulator(dataset_df):
    # Description: construct feature and clean data
    # Output: pandas dataframe
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
    regr = linear_model.LinearRegression()
    regr.fit(train[["R_1000_diff_1", "temp-dew_diff_1", "uv_10m"]], train["visia_diff_1"])
    return (regr.predict(test[["R_1000_diff_1", "temp-dew_diff_1", "uv_10m"]]) + test["visia_shift_1"]) < 3000

EC_DataSet = pd.read_csv("visia.csv")
linear_regresion_gt = BackTest.BackTestGoThrough(
    dataset_df=data_manipulator(EC_DataSet),
    target_col_list=["visia_low"],
    forecast_func=linear_regression_func)

linear_regresion_gt.make_go_through_prediction()
x = linear_regresion_gt.get_train_window()
y = linear_regresion_gt.get_f()

plt.figure()
pd.DataFrame(y, x).plot()
plt.savefig('power.png', dpi=300)
plt.show()