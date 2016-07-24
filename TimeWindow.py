import pandas as pd
from sklearn import linear_model
from sklearn import metrics

import matplotlib.pyplot as plt

import warnings


def warn(*args, **kwargs):
    pass
warnings.warn = warn


def dataset_cleaning(dataset_df):
    # To fill in all missing value with Overall Mean
    # Input: Pandas Dataframe
    # Output: Pandas Dataframe
    for column in dataset_df:
        try:
            dataset_df[column].fillna(dataset_df[column].mean())
        except:
            pass
    return dataset_df


def data_slice_generator(dataset_df, train_window_int, test_window_int):
    # Data subset generator
    # Input: dataset_df pandas dataframe
    #        train_window_int: integer smaller than total observations
    #        test_window_int: integer smaller than total observations - train_window_int
    # Output: Pandas dataframe
        for beginning_int in xrange(0, dataset_df.shape[0]-train_window_int-test_window_int+1):
            yield dataset_df.ix[beginning_int:beginning_int+train_window_int-1+test_window_int, ].reset_index()


def pred_value(dataset_slice_df, train_window_int, test_window_int, feature_col_list, target_col_list, classifier_func):
    # Generate single prediction value
    # Output:Bool
    train = dataset_slice_df.ix[:train_window_int-1, ]
    target_train = train[target_col_list]
    feature_train = train[feature_col_list]

    test = dataset_slice_df.ix[train_window_int+test_window_int-1, ]
    feature_test = test[feature_col_list]
    target_test = test[target_col_list]

    return classifier_func(feature_train, target_train, feature_test, target_test)


def actual_value(dataset_slice_df, train_window_int, test_window_int, target_col_list):
    # Generate single actual value
    # Output: Bool
    return dataset_slice_df.ix[train_window_int+test_window_int-1, target_col_list]


def evaluation(dataset_df,
               train_window_int, test_window_int,
               feature_col_list, target_col_list,
               classifier_func, evaluator_func):
    # Generate the generalization ability score
    # Return int
    pred_list = []
    actual_list = []
    for item in data_slice_generator(dataset_df, train_window_int, test_window_int):
        #try:
            pred_list.append(pred_value(item,
                                        train_window_int, test_window_int,
                                        feature_col_list, target_col_list,
                                        classifier_func))
            actual_list.append(actual_value(item, train_window_int, test_window_int, target_col_list))
        #except ValueError:
            #pass
    return evaluator_func(actual_list, pred_list)


def main_function(dataset_df, test_window_int,
                  classifier_func, evaluator_func,
                  data_manupilater, feature_col_list, target_col_list):
    dataset_clean_df = data_manupilater(dataset_cleaning(dataset_df)).fillna(0)

    evaluation_list = []
    train_window_list = []
    for window in xrange(30, 600+1):
        train_window_list.append(window)
        f_eval, precision_eval, recall_eval = evaluation(dataset_df=dataset_clean_df,
                                                         train_window_int=window,
                                                         test_window_int=test_window_int,
                                                         feature_col_list=feature_col_list,
                                                         target_col_list=target_col_list,
                                                         classifier_func=classifier_func,
                                                         evaluator_func=evaluator_func)
        evaluation_list.append(f_eval)
        print "The evaluation of window %s is done, F1 value is %s, Precision is %s, Recall is %s" \
              % (window, f_eval, precision_eval, recall_eval)
    return train_window_list, evaluation_list


# Execution
# Loading Data
visia_dataset = pd.read_csv("visia.csv")

# 3 Functions must be defined and delivered:
# 1st: the dataset manipulator function, which define target and feature variables
# 2nd: the classifier function, which returns the predict value
# 3rd: he evaluator function, which generate the score of generalization ability


def data_manipulator(dataset_df):
    # Output: pandas dataframe
    dataset_df["visia_diff_1"] = dataset_df["visia"].diff(periods=1)
    dataset_df["visia_shift_1"] = dataset_df["visia"].shift(periods=1)
    dataset_df["visia_low"] = dataset_df["visia"] < 3000

    dataset_df["R_1000_diff_1"] = dataset_df["R_1000"].diff(periods=1)
    dataset_df["temp-dew"] = dataset_df["temperature"]-dataset_df["dewtemp"]
    dataset_df["temp-dew_diff_1"] = dataset_df["temp-dew"].diff(periods=1)
    dataset_df["uv_10m_diff_1"] = dataset_df["uv_10m"].diff(periods=1)
    return dataset_df[["R_1000", "R_1000_diff_1",
                       "temp-dew", "temp-dew_diff_1",
                       "uv_10m", "uv_10m_diff_1",
                       "visia", "visia_diff_1", "visia_shift_1", "visia_low"]]


def linear_regression_func(train_feature, train_target, test_feature, test_target):
    regr = linear_model.LinearRegression()
    regr.fit(train_feature[["R_1000_diff_1", "temp-dew_diff_1", "uv_10m"]], train_target["visia_diff_1"])
    return regr.predict(test_feature[["R_1000_diff_1", "temp-dew_diff_1", "uv_10m"]]) \
           + test_target["visia_shift_1"] < 3000


def evaluation_func(actual_list, pred_list):
    # return metrics.r2_score([item["low"] for item in actual_list], pred_list)
    return metrics.f1_score([item["visia_low"] for item in actual_list], pred_list), \
           metrics.precision_score([item["visia_low"] for item in actual_list], pred_list), \
           metrics.recall_score([item["visia_low"] for item in actual_list], pred_list)

x, y = main_function(dataset_df=visia_dataset,
                     test_window_int=1,
                     classifier_func=linear_regression_func,
                     evaluator_func=evaluation_func,
                     data_manupilater=data_manipulator,
                     feature_col_list=["R_1000", "R_1000_diff_1",
                                       "temp-dew", "temp-dew_diff_1",
                                       "uv_10m", "uv_10m_diff_1"],
                     target_col_list=["visia", "visia_diff_1", "visia_shift_1", "visia_low"])


plt.figure()
pd.DataFrame(y, x).plot()
plt.savefig('power.png', dpi=300)
plt.show()

# Testing on Linux
