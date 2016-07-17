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

def pred_value(dataset_slice_df, train_window_int, test_window_int,target_col_str, classifier_instance_sklearn):
    # Generate single prediction value
    # Output:Bool
    train=dataset_slice_df.ix[:train_window_int-1,]
    target_train = train[target_col_str]
    feature_train=train
    del feature_train[target_col_str]

    feature_test=dataset_slice_df.ix[train_window_int+test_window_int-1,]
    del feature_test[target_col_str]

    clf=classifier_instance_sklearn.fit(feature_train,target_train)
    return clf.predict(feature_test)


def actual_value(dataset_slice_df, train_window_int, test_window_int, target_col_str):
    # Generate single actual value
    # Output: Bool
    return dataset_slice_df.ix[train_window_int+test_window_int-1,target_col_str]

def evaluation(dataset_df,train_window_int, test_window_int,target_col_str, classifier_instance_sklearn, evaluator_sklearn):
    pred_list=[]
    actual_list=[]
    for item in data_slice_generator(dataset_df,train_window_int,test_window_int):
        try:
            pred_list.append(pred_value(item, train_window_int, test_window_int,target_col_str, classifier_instance_sklearn))
            actual_list.append(actual_value(item, train_window_int, test_window_int, target_col_str))
        except:
            pass
    return evaluator_sklearn(actual_list,pred_list)

def main_function(dataset_df,test_window_int,threshold,classifier_instance_sklearn, evaluator_sklearn,data_manupilater):
    target_col_str="low"
    dataset_clean_df=data_manupilater(dataset_cleaning(dataset_df),threshold,target_col_str)

    evaluation_list=[]
    train_window_list=[]
    for window in xrange(30,600+1):
        train_window_list.append(window)
        f_eval=evaluation(dataset_df=dataset_clean_df,
                          train_window_int=window,
                          test_window_int=test_window_int,
                          target_col_str=target_col_str,
                          classifier_instance_sklearn=classifier_instance_sklearn,
                          evaluator_sklearn=evaluator_sklearn)
        evaluation_list.append(f_eval)
        print "The evaluation of window %s is done, F1 value is %s" % (window,f_eval)
    return train_window_list, evaluation_list


#Execution.
visia_dataset=pd.read_csv("visia.csv")

def data_manupilater(dataset_df,threshold,target_col_str):
    # Output: pandas dataframe
    dataset_df[target_col_str]=dataset_df["visia"]<threshold
    dataset_df["R_1000_exp"] = dataset_df["R_1000"].map(lambda x:x**1.5)
    dataset_df["temp-dew"]=dataset_df["temperature"]-dataset_df["dewtemp"]
    dataset_df["sun"]=dataset_df["time"]%100-12
    return dataset_df[["R_1000_exp","temp-dew","uv_10m",target_col_str]]

from sklearn.linear_model import LogisticRegression
clf_LogitRegression = LogisticRegression()

from sklearn import metrics
x,y = main_function(dataset_df=visia_dataset,
                    test_window_int=1,
                    threshold=1000,
                    classifier_instance_sklearn=clf_LogitRegression,
                    evaluator_sklearn=metrics.f1_score,
                    data_manupilater=data_manupilater)

import matplotlib.pyplot as plt
plt.figure()
pd.DataFrame(y,x).plot()
plt.savefig('power.png', dpi=300)
plt.show()