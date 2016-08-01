"""
This script design and test algorithm based on actual meteorology data in HangZhou, China
It is for the visibility forecasting project initiated by William Wei (weichen.211@qq.com)
Author: Ben Wang (lbenwong@live.cn)
"""
import pandas as pd
import BackTest

from sklearn import linear_model

import warnings


def warn(*args, **kwargs):
    pass
warnings.warn = warn


def data_manipulator(dataset_df):
    # Description: construct feature and clean data
    # Output: pandas dataframe
    dataset_df["VisiA_diff_1"] = dataset_df["VisiA"].diff(periods=1)
    dataset_df["VisiA_shift_1"] = dataset_df["VisiA"].shift(periods=1)
    dataset_df["VisiA_low"] = dataset_df["VisiA"] < 3000

    dataset_df["RelHumidity_diff_1"] = dataset_df["RelHumidity"].diff(periods=1)
    dataset_df["Temp-DewTemp"] = dataset_df["SurfaceTemp"] - dataset_df["DewTemp"]
    dataset_df["Temp-DewTemp_diff_1"] = dataset_df["Temp-DewTemp"].diff(periods=1)
    dataset_df["WindVelocity_diff_1"] = dataset_df["WindVelocity"].diff(periods=1)

    dataset_df = dataset_df[["RelHumidity", "RelHumidity_diff_1",
                             "Temp-DewTemp", "Temp-DewTemp_diff_1",
                             "WindVelocity", "WindVelocity_diff_1",
                             "VisiA", "VisiA_diff_1",
                             "VisiA_shift_1", "VisiA_low"]]

    for column in dataset_df:
        dataset_df[column].fillna(dataset_df[column].mean())
    dataset_df = dataset_df.fillna(0)

    return dataset_df[["RelHumidity", "RelHumidity_diff_1",
                       "Temp-DewTemp", "Temp-DewTemp_diff_1",
                       "WindVelocity", "WindVelocity_diff_1",
                       "VisiA", "VisiA_diff_1",
                       "VisiA_shift_1", "VisiA_low"]]


def classifier_func(train, test):
    # Description: The core modelling module
    # input: pandas data frame split into train set and test set
    # output: single test result
    regr = linear_model.LinearRegression()
    regr.fit(train[["RelHumidity_diff_1", "Temp-DewTemp_diff_1", "WindVelocity_diff_1"]], train["VisiA_diff_1"])
    result = regr.predict(test[["RelHumidity_diff_1", "Temp-DewTemp_diff_1", "WindVelocity_diff_1"]])
    base = test["VisiA_shift_1"].reset_index()["VisiA_shift_1"]
    return (base[0] + result[0] + result[1] + result[2]) < 3000


DataSet = pd.read_csv("Data/HZ_2015.csv")
Model_BackTest = BackTest.BackTestGoThrough(
    dataset_df=data_manipulator(DataSet),
    target_col_list=["VisiA_low"],
    forecast_func=classifier_func)

Model_BackTest.make_go_through_prediction(min_window=30, max_window=1000, test_period_int=3) \
              .evaluation_chart(filename="HZ_Score (Predict 3)")

