from dataload.load import ModelDataset
from sklearn.preprocessing import *
from scipy.stats import *
import pandas as pd
import numpy as np
import datetime

""" Common preprocessing """
def before_making_lookback(dataset):
    dataset = dataset.drop(['seawave'], axis=1)
    dataset = dataset.dropna(axis=0)

    x_data = dataset.iloc[:, :-1]
    y_data = dataset.iloc[:, -1]

    # y_data = y_data.apply(np.sqrt)

    # tz_dummy = pd.get_dummies(x_data['tgt_tz'])
    # tz_dummy.columns = ['tz_0', 'tz_1', 'tz_2', 'tz_3', 'tz_4', 'tz_5', 'tz_6', 'tz_7']
    # x_data = pd.concat([tz_dummy, x_data], axis=1)

    # one-hot encoding
    raintype = pd.get_dummies(x_data['raintype'], prefix='raintype')
    skystatus = pd.get_dummies(x_data['skystatus'], prefix='skystatus')

    x_data = pd.concat([x_data, raintype], axis=1)
    x_data = pd.concat([x_data, skystatus], axis=1)

    x_data = x_data.drop(['raintype'], axis=1)
    x_data = x_data.drop(['skystatus'], axis=1)
    return x_data, y_data

def generating_lookback(x_data, y_data, label, lookbackshift, baselookback):
    for shift in range(lookbackshift):
        globals()['generation_{}'.format(shift)] = y_data.shift(shift + 2)
        x_data = pd.concat([x_data, globals()['generation_{}'.format(shift)]], axis=1)
        x_data = x_data.rename(columns={label: 'y_t{}h'.format(baselookback + (shift * 3))})
    return x_data

def after_making_lookback(x_data, y_data):
    dataset = pd.concat([x_data, y_data], axis=1)
    dataset = dataset.dropna(axis=0)
    dataset = dataset.reset_index(drop=True)
    return dataset

def before_scaling(dataset):
    train_set = dataset[dataset['tgt_Y'] != 2017]
    test_set = dataset[dataset['tgt_Y'] == 2017]

    test_set_time_dummy = test_set[['tgt_Y', 'tgt_M', 'tgt_D', 'tgt_tz']]
    test_set_time_dummy = test_set_time_dummy.reset_index(drop=True)

    for i in range(len(test_set_time_dummy)):
        d = datetime.date(test_set_time_dummy.loc[i, 'tgt_Y'], test_set_time_dummy.loc[i, 'tgt_M'],
                          test_set_time_dummy.loc[i, 'tgt_D'])
        t = datetime.time(int(test_set_time_dummy.loc[i, 'tgt_tz'] * 3))
        dt = datetime.datetime.combine(d, t)
        test_set_time_dummy.loc[i, 'tgt_datetime'] = dt

    train_set = train_set.drop(['tgt_Y', 'tgt_M', 'tgt_D'], axis=1)
    test_set = test_set.drop(['tgt_Y', 'tgt_M', 'tgt_D'], axis=1)

    train_x, train_y = train_set.iloc[:, :-1], train_set.iloc[:, -1]
    column_list = train_x.columns
    test_x, test_y = test_set.iloc[:, :-1], test_set.iloc[:, -1]
    return test_set_time_dummy, column_list, train_x, train_y, test_x, test_y


"""Generating historical wind power generation data"""
class ModelPreprocess(ModelDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def data_load(self):
        self.dataset.drop(self.dataset.tail(1).index, inplace=True)
        self.dataset = self.dataset.reset_index(drop=True)
        print(self.dataset.info())

    # Not generating historical power generation data
    def data_processing_h0(self):
        x_data, y_data = before_making_lookback(self.dataset)
        self.dataset = after_making_lookback(x_data, y_data)
        return self.dataset

    # Generating historical power generation data (before 3 hours)
    def data_processing_h1(self, label):
        x_data, y_data = before_making_lookback(self.dataset)

        LOOKBACKSHIFT = 1
        BASELOOKBACK = 2

        x_data = generating_lookback(x_data, y_data, label, LOOKBACKSHIFT, BASELOOKBACK)
        self.dataset = after_making_lookback(x_data, y_data)
        return self.dataset

    # Generating historical power generation data (before 6 hours)
    def data_processing_h2(self, label):
        x_data, y_data = before_making_lookback(self.dataset)

        LOOKBACKSHIFT = 2
        BASELOOKBACK = 2

        x_data = generating_lookback(x_data, y_data, label, LOOKBACKSHIFT, BASELOOKBACK)
        self.dataset = after_making_lookback(x_data, y_data)
        return self.dataset

    # Generating historical power generation data (before 9 hours)
    def data_processing_h3(self, label):
        x_data, y_data = before_making_lookback(self.dataset)

        LOOKBACKSHIFT = 3
        BASELOOKBACK = 2

        x_data = generating_lookback(x_data, y_data, label, LOOKBACKSHIFT, BASELOOKBACK)
        self.dataset = after_making_lookback(x_data, y_data)
        return self.dataset

    # Generating historical power generation data (before 12 hours)
    def data_processing_h4(self, label):
        x_data, y_data = before_making_lookback(self.dataset)

        LOOKBACKSHIFT = 4
        BASELOOKBACK = 2

        x_data = generating_lookback(x_data, y_data, label, LOOKBACKSHIFT, BASELOOKBACK)
        self.dataset = after_making_lookback(x_data, y_data)
        return self.dataset

    # Generating historical power generation data (before 15 hours)
    def data_processing_h5(self, label):
        x_data, y_data = before_making_lookback(self.dataset)

        LOOKBACKSHIFT = 5
        BASELOOKBACK = 2

        x_data = generating_lookback(x_data, y_data, label, LOOKBACKSHIFT, BASELOOKBACK)
        self.dataset = after_making_lookback(x_data, y_data)
        return self.dataset

    # Dataset scaling and split train/test dataset
    def split_dataset(self, dataset):
        test_set_time_dummy, column_list, train_x, train_y, test_x, test_y = before_scaling(dataset)

        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))

        train_x = scaler_x.fit_transform(train_x)
        test_x = scaler_x.transform(test_x)

        train_y = scaler_y.fit_transform(train_y.values.reshape(-1, 1))
        test_y = scaler_y.transform(test_y.values.reshape(-1, 1))

        # Target variable : not scaling
        # train_y = train_y.values.reshape(-1, 1)
        # test_y = test_y.values.reshape(-1, 1)
        return scaler_x, scaler_y, train_x, train_y, test_x, test_y, test_set_time_dummy, column_list

    # Split train/test dataset not scaling
    def split_dataset_none_scaling(self, dataset):
        test_set_time_dummy, column_list, train_x, train_y, test_x, test_y = before_scaling(dataset)

        train_x = train_x.to_numpy()
        test_x = test_x.to_numpy()

        train_y = train_y.values.reshape(-1, 1)
        test_y = test_y.values.reshape(-1, 1)
        return train_x, train_y, test_x, test_y, test_set_time_dummy, column_list