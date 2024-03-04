from sklearn.linear_model import LinearRegression


class LRegression:
    def __init__(self, model_param: dict):
        self.fit_intercept = model_param['fit_intercept']
        self.normalize = model_param['normalize']

    def mlr_regressor(self, x, y):
        mlr = LinearRegression(fit_intercept=self.fit_intercept, normalize=self.normalize)
        model = mlr.fit(x, y)

        y_fit = model.predict()
        return model
