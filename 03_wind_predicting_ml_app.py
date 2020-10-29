from datapreprocess.datasetformodel import *
from mlmodel.mlmodel import *
from featureimportance.feature_interpretation import *
from datavisualization.visualization import *

power_plants = ['hk1', 'hk2', 'ss']


def ml_model_operate(dataset, datset_target, plant, his):
    # # Dataset splitting
    scaler_x, scaler_y, train_x_scaled, train_y_scaled, test_x_scaled,\
    test_y_scaled, time_dummy_set, column_list = ModelPreprocess(dataset).split_dataset(datset_target)

    # Model fitting - Linear regression models
    mlr_model = MlModel(scaler_x, scaler_y, train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled,
                        time_dummy_set).mlr_regressor()
    MlVisualization(scaler_x, scaler_y, train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled, time_dummy_set,
                    mlr_model).ml_predict(f'{plant}', f'h{his}', f'{plant} Generation_MLR H{his}')

    pls_model = MlModel(scaler_x, scaler_y, train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled,
                        time_dummy_set).pls_regressor()
    MlVisualization(scaler_x, scaler_y, train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled, time_dummy_set,
                    pls_model).ml_predict(f'{plant}', f'h{his}', f'{plant} Generation_PLS H{his}')

    elas_model = MlModel(scaler_x, scaler_y, train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled,
                         time_dummy_set).elasticnet_regressor(
        column_list, f'{plant} Generation_ElasticNet H{his}')
    MlVisualization(scaler_x, scaler_y, train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled, time_dummy_set,
                    elas_model).ml_predict(f'{plant}', f'h{his}', f'{plant} Generation_ElasticNet H{his}')

    #Gamma, Tweedie distribution regression model
    # gamma_model = MlModel(scaler_x, scaler_y, train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled,
    #                     time_dummy_set).gamma_regressor()
    # MlVisualization(scaler_x, scaler_y, train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled, time_dummy_set,
    #                 gamma_model).ml_predict(f'{plant}', f'h{his}', f'{plant} Generation_Gamma H{his}')

    # # Model fitting - Tweedie distribution regression model
    # tweedie_model = MlModel(scaler_x, scaler_y, train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled,
    #                      time_dummy_set).tweedie_regressor()
    # MlVisualization(scaler_x, scaler_y, train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled, time_dummy_set,
    #                 tweedie_model).ml_predict(f'{plant}', f'h{his}', f'{plant} Generation_Tweedie_001 H{his}')


    # Model fitting - Support vector machine regression models
    svr_model = MlModel(scaler_x, scaler_y, train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled,
                        time_dummy_set).svr_regressors()
    MlVisualization(scaler_x, scaler_y, train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled, time_dummy_set,
                    svr_model).ml_predict(f'{plant}', f'h{his}', f'{plant} Generation_SVR H{his}')



    # Model fitting - Tree based regression model
    train_x, train_y, test_x, test_y, test_set_time_dummy, column_list = ModelPreprocess(
        dataset).split_dataset_none_scaling(datset_target)

    dt_model = MlModel(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set).dt_regressor()
    MlVisualization(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set,
                    dt_model).ml_predict_tree_based(f'{plant}', f'h{his}', f'{plant} Generation_DT H{his}')

    # Ensemble Model fitting - Tree based regression ensemble models
    rf_model = MlModel(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set).rf_regressor(column_list,
                                                                                                          f'{plant} Generation_RF H{his}')
    MlVisualization(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set,
                    rf_model).ml_predict_tree_based(f'{plant}', f'h{his}', f'{plant} Generation_RF H{his}')

    bag_model = MlModel(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set).bagging_regressor()
    MlVisualization(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set,
                    bag_model).ml_predict_tree_based(f'{plant}', f'h{his}', f'{plant} Generation_Bag H{his}')

    ada_model = MlModel(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set).adaboost_regressor()
    MlVisualization(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set,
                    ada_model).ml_predict_tree_based(f'{plant}', f'h{his}', f'{plant} Generation_Ada H{his}')

    gbr_model = MlModel(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set).gbr_regressors()
    MlVisualization(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set,
                    gbr_model).ml_predict_tree_based(f'{plant}', f'h{his}', f'{plant} Generation_GBR H{his}')

    xgb_model = MlModel(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set).xgboost_regressor()
    MlVisualization(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set,
                    xgb_model).ml_predict_tree_based(f'{plant}', f'h{his}', f'{plant} Generation_XGB H{his}')

    lgbm_model = MlModel(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set).lightgbm_regressor()
    MlVisualization(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set,
                    lgbm_model).ml_predict_tree_based(f'{plant}', f'h{his}', f'{plant} Generation_LGBM H{his}')


def feature_importance_xgboost(dataset, datset_target, plant, his):
    # # Dataset splitting
    scaler_x, scaler_y, train_x_scaled, train_y_scaled, test_x_scaled, \
    test_y_scaled, time_dummy_set, column_list = ModelPreprocess(dataset).split_dataset(datset_target)

    # Model fitting - Tree based regression model
    train_x, train_y, test_x, test_y, test_set_time_dummy, column_list = ModelPreprocess(
        dataset).split_dataset_none_scaling(datset_target)

    xgb_model = FIModel(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set).xgboost_importances(
        column_list, f'{plant} Generation_Xgboost H{his}')
    MlVisualization(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set,
                    xgb_model).ml_predict_tree_based(f'{plant}', f'h{his}', f'{plant} Generation_Xgboost_ H{his}')

if __name__ == '__main__':
    for plant in power_plants:
        globals()[f'{plant}_dataset_path'] = f'./dataset/mergeddata/wind_weather_forecast_{plant}_4hours.csv'
        globals()[f'{plant}_dataset'] = ModelDataset().dataset_load(globals()[f'{plant}_dataset_path'])
        ModelPreprocess(globals()[f'{plant}_dataset']).data_load()

        for his in range(6):
            if his == 0:
                globals()[f'{plant}_dataset_target'] = ModelPreprocess(globals()[f'{plant}_dataset']).data_processing_h0()
                # ml_model_operate(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)
                feature_importance_xgboost(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)

            elif his == 1:
                globals()[f'{plant}_dataset_target'] = ModelPreprocess(globals()[f'{plant}_dataset']).data_processing_h1(
                    f'{plant}')
                # ml_model_operate(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)
                feature_importance_xgboost(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)

            elif his == 2:
                globals()[f'{plant}_dataset_target'] = ModelPreprocess(globals()[f'{plant}_dataset']).data_processing_h2(
                    f'{plant}')
                # ml_model_operate(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)
                feature_importance_xgboost(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)

            elif his == 3:
                globals()[f'{plant}_dataset_target'] = ModelPreprocess(globals()[f'{plant}_dataset']).data_processing_h3(
                    f'{plant}')
                # ml_model_operate(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)
                feature_importance_xgboost(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)

            elif his == 4:
                globals()[f'{plant}_dataset_target'] = ModelPreprocess(globals()[f'{plant}_dataset']).data_processing_h4(
                    f'{plant}')
                # ml_model_operate(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)
                feature_importance_xgboost(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)

            else:
                globals()[f'{plant}_dataset_target'] = ModelPreprocess(globals()[f'{plant}_dataset']).data_processing_h5(
                    f'{plant}')
                # ml_model_operate(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)
                feature_importance_xgboost(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)