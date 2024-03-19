from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.cross_decomposition import *
from sklearn.svm import *
from sklearn import ensemble
from sklearn.ensemble import *

from lightgbm import LGBMRegressor
from numpy.random import seed
from datapreprocess.datasetformodel import ModelPreprocess

import xgboost as xgb
import pandas as pd

seed(1)


class FIModel(ModelPreprocess):
    def __init__(self, scaler_x, scaler_y, train_x, train_y, test_x, test_y, test_set_time_dummy):
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.test_set_time_dummy = test_set_time_dummy

    def xgboost_importances(self, column_list, title):
        xgbr = xgb.XGBRegressor(objective='reg:squarederror')
        model = xgbr.fit(self.train_x, self.train_y)

        feat_importances = pd.Series(model.feature_importances_, index=column_list)
        feat_importances = feat_importances.to_frame(name='Importance')
        feat_importances.to_csv(f'./predictionresult/xgboost/{title}.csv')
        feat_importances.plot(kind='barh', title='Feature Importances').invert_yaxis()
        return model