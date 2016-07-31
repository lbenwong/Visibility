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
    # TODO
    pass


def classifier_func(train, test):
    # TODO
    pass


DataSet = pd.read_csv("HZ_2015.csv")
Model_BackTest = BackTest.BackTestGoThrough(
    dataset_df=data_manipulator(DataSet),
    target_col_list=[],  # TODO
    forecast_func=classifier_func)

Model_BackTest.make_go_through_prediction(min_window=30, max_window=500) \
              .evaluation_chart()

