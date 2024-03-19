from datapreprocess.datasetformodel import *
from mlmodel.mlmodel import *
from datavisualization.visualization import *

dnn_parameter = {'batch_size': 32, 'activation': 'relu', 'nuerons': 64, 'dropout': 0, 'output_activation': 'linear',
                     'kernel_initializer': 'he_uniform', 'n_hidden_layer': 4, 'input_kernel_regularizer': 0.0001,
                     'dense_kernel_regularizer': 0.0001}
power_plants = ['hk1', 'hk2', 'ss']


def dnn_model_operate(dataset, datset_target, plant, his):
    # Dataset splitting
    scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set, column_list = ModelPreprocess(
        dataset).split_dataset(datset_target)

    # Model fitting
    model = DnnModel(scaler_x, scaler_y, train_x, train_y, test_x, test_y,
                     time_dummy_set).model_preformance_and_result(
        batch_size=dnn_parameter['batch_size'],
        activation=dnn_parameter['activation'],
        nuerons=dnn_parameter['nuerons'],
        dropout=dnn_parameter['dropout'],
        output_activation=dnn_parameter['output_activation'],
        kernel_initializer=dnn_parameter['kernel_initializer'],
        n_hidden_layer=dnn_parameter['n_hidden_layer'],
        input_kernel_regularizer=dnn_parameter['input_kernel_regularizer'],
        dense_kernel_regularizer=dnn_parameter['dense_kernel_regularizer']
    )

    # Prediction visualization
    DlVisualization(scaler_x, scaler_y, train_x, train_y, test_x, test_y, time_dummy_set,
                    model).dl_predict(f'{plant}', f'h{his}', f'{plant} Generation_MLP H{his}')

if __name__ == '__main__':
    for plant in power_plants:
        globals()[f'{plant}_dataset_path'] = f'./dataset/mergeddata/wind_weather_forecast_{plant}_4hours.csv'
        globals()[f'{plant}_dataset'] = ModelDataset().dataset_load(globals()[f'{plant}_dataset_path'])
        ModelPreprocess(globals()[f'{plant}_dataset']).data_load()

        for his in range(6):
            if his == 0:
                globals()[f'{plant}_dataset_target'] = ModelPreprocess(globals()[f'{plant}_dataset']).data_processing_h0()
                dnn_model_operate(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)

            elif his == 1:
                globals()[f'{plant}_dataset_target'] = ModelPreprocess(globals()[f'{plant}_dataset']).data_processing_h1(
                    f'{plant}')
                dnn_model_operate(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)

            elif his == 2:
                globals()[f'{plant}_dataset_target'] = ModelPreprocess(globals()[f'{plant}_dataset']).data_processing_h2(
                    f'{plant}')
                dnn_model_operate(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)

            elif his == 3:
                globals()[f'{plant}_dataset_target'] = ModelPreprocess(globals()[f'{plant}_dataset']).data_processing_h3(
                    f'{plant}')
                dnn_model_operate(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)

            elif his == 4:
                globals()[f'{plant}_dataset_target'] = ModelPreprocess(globals()[f'{plant}_dataset']).data_processing_h4(
                    f'{plant}')
                dnn_model_operate(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)

            else:
                globals()[f'{plant}_dataset_target'] = ModelPreprocess(globals()[f'{plant}_dataset']).data_processing_h5(
                    f'{plant}')
                dnn_model_operate(globals()[f'{plant}_dataset'], globals()[f'{plant}_dataset_target'], plant, his)