def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import math

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
        for beginning_int in xrange(0,dataset_df.shape[0]-train_window_int-test_window_int+1):
            yield dataset_df.ix[beginning_int:beginning_int+train_window_int-1+test_window_int,].reset_index()

def pred_value(dataset_slice_df, train_window_int, test_window_int,feature_col_list,target_col_str, classifier_func):
    # Generate single prediction value
    # Output:Bool
    train=dataset_slice_df.ix[:train_window_int-1,]
    target_train = train[target_col_str]
    feature_train = train[feature_col_list]

    feature_test = dataset_slice_df.ix[train_window_int+test_window_int-1,]
    feature_test = feature_test[feature_col_list]

    return classifier_func(feature_train,target_train,feature_test)

def actual_value(dataset_slice_df, train_window_int, test_window_int, target_col_str):
    # Generate single actual value
    # Output: Bool
    return dataset_slice_df.ix[train_window_int+test_window_int-1,target_col_str]

def evaluation(dataset_df,train_window_int, test_window_int,feature_col_list,target_col_str, classifier_func, evaluator_func):
    pred_list=[]
    actual_list=[]
    for item in data_slice_generator(dataset_df,train_window_int,test_window_int):
        try:
            pred_list.append(pred_value(item, train_window_int, test_window_int,feature_col_list,target_col_str, classifier_func))
            actual_list.append(actual_value(item, train_window_int, test_window_int, target_col_str))
        except:
            pass
    return evaluator_func(actual_list,pred_list)

def main_function(dataset_df,test_window_int,classifier_func, evaluator_func,data_manupilater,feature_col_list,target_col_str):
    dataset_clean_df=data_manupilater(dataset_cleaning(dataset_df),target_col_str)

    evaluation_list=[]
    train_window_list=[]
    for window in xrange(30,600+1):
        train_window_list.append(window)
        f_eval=evaluation(dataset_df=dataset_clean_df,
                          train_window_int=window,
                          test_window_int=test_window_int,
                          feature_col_list=feature_col_list,
                          target_col_str=target_col_str,
                          classifier_func=classifier_func,
                          evaluator_func=evaluator_func)
        evaluation_list.append(f_eval)
        print "The evaluation of window %s is done, F1 value is %s" % (window,f_eval)
    return train_window_list, evaluation_list


#Execution
#Loading Data
visia_dataset=pd.read_csv("visia.csv")

# 3 Functions must be defined and delivered:
# 1st: the dataset manupulater function, which define target and feature variables
# 2nd: the classifier function, which returns the predict value
# 3rd: he evaluator function, which generate the score of generalization ability

def data_manupilater(dataset_df,target_col_str):
    # Output: pandas dataframe
    dataset_df[target_col_str]=dataset_df["visia"]<3000
    dataset_df["R_1000_exp"] = dataset_df["R_1000"].map(lambda x:x**1.5)
    dataset_df["temp-dew"]=dataset_df["temperature"]-dataset_df["dewtemp"]
    dataset_df["sun"]=dataset_df["time"]%100-12
    return dataset_df[["R_1000_exp","temp-dew","uv_10m",target_col_str]]

from sklearn.linear_model import LogisticRegression
def clf_LogitRegression_func(train_feature,train_target,test_feature):
    clf_LogitRegression=LogisticRegression()
    clf_LogitRegression_Model=clf_LogitRegression.fit(train_feature, train_target)
    return clf_LogitRegression_Model.predict(test_feature)

from sklearn import metrics
x,y = main_function(dataset_df = visia_dataset,
                    test_window_int = 1,
                    classifier_func = clf_LogitRegression_func,
                    evaluator_func = metrics.f1_score,
                    data_manupilater = data_manupilater,
                    feature_col_list = ["R_1000_exp","temp-dew","uv_10m"],
                    target_col_str = "low")

import matplotlib.pyplot as plt
plt.figure()
pd.DataFrame(y,x).plot()
plt.savefig('power.png', dpi=300)
plt.show()