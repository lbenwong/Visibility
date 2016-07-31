"""
This module provides a back test framework of time series forecasting
It is for the visibility forecasting project initiated by William Wei (weichen.211@qq.com)
Author: Ben Wang (lbenwong@live.cn)
"""

from sklearn import metrics
import matplotlib.pylab as plt


def _default_evaluation(actual_list, prediction_list):
    return metrics.f1_score(y_true=actual_list, y_pred=prediction_list), \
           metrics.precision_score(y_true=actual_list, y_pred=prediction_list), \
           metrics.recall_score(y_true=actual_list, y_pred=prediction_list)


class BackTest(object):
    def __init__(self, dataset_df, target_col_list, forecast_func, evaluation_func=_default_evaluation):
        """
        Description: This module generate prediction list given the dataframe and forecast function;
                     prediction evaluation is available given the target_col_list and evaluation function.

        :param dataset_df: a pandas data frame including both feature and target
        :param target_col_list: a list indicating the column of actual target value in the data frame (should be 1 col)
        :param forecast_func: function object that take in data frame and returns single int/bool forecast result
        :param evaluation_func: function object that returns f-value, precision and recall

        Attribute: No attribute is exposed
        Method:
        1. make_prediction(train_window_int, test_window_int)
                Parameters:
                    train_window_int Type: Integer Description: Indicating no. of period used for model training
                    test_window_int Type: Integer Description: Indicating no. of period away to predict
                Return: self

        2. get_prediction_list Para: No; Return: A list of Prediction
        3. get_actual_list Para: No; Return: A list of corresponding actual value
        4. get_evaluation Para: No; Return: 3 float of evaluation result, namely, f-value, precision, recall
        """
        self._dataset = dataset_df
        self._target_col = target_col_list
        self._forecaster = forecast_func
        self._evaluator = evaluation_func

        self._train = None
        self._test = None

        self._prediction = []
        self._actual = []

    def _generate_data_slice(self, train_window_int, test_window_int):
        for beginning_int in xrange(0, self._dataset.shape[0] - train_window_int - test_window_int + 1):
            yield self._dataset.ix[beginning_int:beginning_int + train_window_int - 1 + test_window_int, ].reset_index()

    def _separate_train_test(self, dataset_slice_df, train_window_int, test_window_int):
        self._train = dataset_slice_df.ix[:train_window_int - 1, ]
        self._test = dataset_slice_df.ix[train_window_int+test_window_int - 1, ]

    def _get_prediction(self):
        return self._forecaster(self._train, self._test)

    def _get_actual_value(self):
        return self._test.ix[self._target_col][0]

    def make_prediction(self, train_window_int, test_window_int):
        for item in self._generate_data_slice(train_window_int, test_window_int):
            try:
                self._separate_train_test(item, train_window_int, test_window_int)
                self._prediction.append(self._get_prediction())
                self._actual.append(self._get_actual_value())
            except ValueError:
                pass
        return self

    def get_prediction_list(self):
        return self._prediction

    def get_actual_list(self):
        return self._actual

    def get_evaluation(self):
        return self._evaluator(self._actual, self._prediction)


class BackTestGoThrough(BackTest):
    def __init__(self, dataset_df, target_col_list, forecast_func, evaluation_func=_default_evaluation):
        """
        Description: This module generate prediction evaluation value in all window of given range

        :param dataset_df: a pandas data frame including both feature and target
        :param target_col_list: a list indicating the column of actual target value in the data frame (should be 1 col)
        :param forecast_func: function object that take in data frame and returns single int/bool forecast result
        :param evaluation_func: function object that returns f-value, precision, recall
        """
        super(BackTestGoThrough, self).__init__(
            dataset_df=dataset_df,
            target_col_list=target_col_list,
            forecast_func=forecast_func,
            evaluation_func=evaluation_func
        )

        self._f_list = []
        self._precision_list = []
        self._recall_list = []
        self._train_window_list = []

    def make_go_through_prediction(self, min_window=100, max_window=400, test_period_int=1):
        for window in xrange(min_window, max_window + 1):
            self.make_prediction(window, test_period_int)
            f_eval, precision_eval, recall_eval = self.get_evaluation()
            self._train_window_list.append(window)
            self._f_list.append(f_eval)
            self._precision_list.append(precision_eval)
            self._recall_list.append(recall_eval)
            print "The evaluation of window %s is done, F1 value is %s, Precision is %s, Recall is %s" \
                  % (window, f_eval, precision_eval, recall_eval)
        return self

    def get_train_window(self):
        return self._train_window_list

    def get_f(self):
        return self._f_list

    def get_precision(self):
        return self._precision_list

    def get_recall(self):
        return self._recall_list

    def evaluation_chart(self, save=1, filename="Score.png", dpi=300):
        plt.figure()
        plt.plot(self._train_window_list, self._f_list, label="F Value", color="red")
        plt.plot(self._train_window_list, self._precision_list, label="Precision", color="green")
        plt.plot(self._train_window_list, self._recall_list, label="recall", color="blue")
        plt.xlabel("Length of Window")
        plt.ylabel("Score")
        plt.title("Red: F-Value; Green: Precision; Blue: Recall")
        if save == 1:
            plt.savefig(filename, dpi=dpi)
        plt.show()
