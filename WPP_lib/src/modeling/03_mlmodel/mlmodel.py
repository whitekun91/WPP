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


class MlModel(ModelPreprocess):
    def __init__(self, scaler_x, scaler_y, train_x, train_y, test_x, test_y, test_set_time_dummy):
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.test_set_time_dummy = test_set_time_dummy

    def mlr_regressor(self):
        mlr = LinearRegression(fit_intercept=True, normalize=False)
        model = mlr.fit(self.train_x, self.train_y)
        return model

    def pls_regressor(self):
        pls = PLSRegression()
        model = pls.fit(self.train_x, self.train_y)
        return model

    def elasticnet_regressor(self, column_list, title):
        en = ElasticNetCV(cv=3, random_state=0)
        model = en.fit(self.train_x, self.train_y.ravel())
        coef_dict_baseline = {}
        for coef, feat in zip(model.coef_, column_list):
            coef_dict_baseline[feat] = coef
        elasticnet_coefficient = pd.DataFrame.from_dict(coef_dict_baseline, orient='index')
        elasticnet_coefficient.to_csv(f'./predictionresult/elasticnet/{title}_coefficient.csv')
        return model

    def dt_regressor(self):
        dt = DecisionTreeRegressor()
        model = dt.fit(self.train_x, self.train_y)
        return model

    def svr_regressors(self):
        svr = LinearSVR(max_iter=10000)
        model = svr.fit(self.train_x, self.train_y.ravel())
        return model

    def rf_regressor(self, column_list, title):
        rf = RandomForestRegressor(n_estimators=100, criterion='mse')
        model = rf.fit(self.train_x, self.train_y.ravel())

        feat_importances = pd.Series(model.feature_importances_, index=column_list)
        feat_importances = feat_importances.to_frame(name='Importance')
        feat_importances.to_csv(f'./predictionresult/randomforest/{title}.csv')
        feat_importances.plot(kind='barh', title='Feature Importances').invert_yaxis()
        return model

    def bagging_regressor(self):
        bg = BaggingRegressor()
        model = bg.fit(self.train_x, self.train_y.ravel())
        return model

    def adaboost_regressor(self):
        ada = AdaBoostRegressor()
        model = ada.fit(self.train_x, self.train_y.ravel())
        return model

    def gbr_regressors(self):
        params = {}
        gbr = ensemble.GradientBoostingRegressor(**params)
        model = gbr.fit(self.train_x, self.train_y.ravel())
        return model

    def xgboost_regressor(self):
        # objective='reg:tweedie'
        # objective='reg:squarederror'
        xgbr = xgb.XGBRegressor(objective='reg:squarederror')
        model = xgbr.fit(self.train_x, self.train_y)
        return model

    def lightgbm_regressor(self):
        lgbr = LGBMRegressor(objective='regression')
        model = lgbr.fit(self.train_x, self.train_y.ravel())
        return model