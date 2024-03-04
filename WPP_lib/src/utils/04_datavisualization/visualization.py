from dlmodel.dnnmodel import DnnModel
from mlmodel.mlmodel import MlModel
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import csv
import numpy as np
import pandas as pd
import keras

import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as po

# Report Code
def regression_report(y_pred, y_true, path, his, title):
    r2 = r2_score(y_pred=y_pred, y_true=y_true)
    mse = mean_squared_error(y_pred=y_pred, y_true=y_true)
    rmse = mean_squared_error(y_pred=y_pred, y_true=y_true) ** 0.5
    mae = mean_absolute_error(y_pred=y_pred, y_true=y_true)

    print(title)
    print("R2:", r2)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print(" ")

    data = [title, r2, mse, rmse, mae]

    # min_value = np.min((y_pred, y_true))
    # max_value = np.max((y_pred, y_true))

    plt.figure(figsize=(20, 7))
    plt.title(title, fontsize=16)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Power generation [kW]', fontsize=16)
    plt.plot(y_true, marker=",", label='True Value')
    plt.plot(y_pred, ls="--", marker=",", label='Predict Value')
    plt.legend(loc='upper left')
    plt.savefig(f"./predictionresult/{path}/{his}/{title}.png")
    plt.clf()
    plt.close()

    with open("./predictionresult/model_performance.csv", 'a', newline="") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(data)
    csvFile.close()

# Common reshaping result
def reshape_result_and_saving(test_set_time_dummy, yhat, test_y, path, his, title):
    result_yhat = yhat.reshape(-1, )
    result_testy = test_y.reshape(-1, )

    result_set = [result_yhat, result_testy]
    result_set = np.transpose(result_set)
    result = pd.DataFrame(result_set, columns=['predict_value', 'actual_value'])
    result = result.reset_index(drop=True)

    # result.loc[:, 'predict_value'] = result.loc[:, 'predict_value'].apply(np.sqrt)
    # result.loc[:, 'actual_value'] = result.loc[:, 'actual_value'].apply(np.sqrt)

    test_set_time_dummy = test_set_time_dummy.reset_index(drop=True)

    result = pd.concat([test_set_time_dummy, result], axis=1)
    result = result.drop(['tgt_Y', 'tgt_M', 'tgt_D', 'tgt_tz'], axis=1)
    result.to_csv(f'./predictionresult/{path}/{his}/{title}.csv', index=False)
    regression_report(yhat, test_y, path, his, title)
    keras.backend.clear_session()

# Resulting process
def predicting_process(scaler_y, test_set_time_dummy, yhat, test_y, path, his, title):
    yhat = scaler_y.inverse_transform(yhat.reshape(-1, 1))
    test_y = scaler_y.inverse_transform(test_y.reshape(-1, 1))
    reshape_result_and_saving(test_set_time_dummy, yhat, test_y, path, his, title)

def predicting_process_tree_based(test_set_time_dummy, yhat, test_y, path, his, title):
    reshape_result_and_saving(test_set_time_dummy, yhat, test_y, path, his, title)


def scatterplot_prediction_residual_common(dataset, x, fig, name, model, column_1, column_2):
    fig.add_trace(
        go.Scatter(x=dataset['actual_value'], y=dataset['{}_predict'.format(name)], mode="markers",
                   name="{}".format(name), marker=dict(color='Grey', opacity=0.5)), row=(model + 1), col=column_1)
    fig.add_trace(
        go.Scatter(x=x, y=x, mode='lines', name='lines', line=dict(color='grey', width=1, dash='dash')),
        row=(model + 1), col=column_1)

    fig.add_trace(
        go.Scatter(x=dataset['{}_predict'.format(name)], y=dataset['{}_residual'.format(name)],
                   mode="markers",
                   name="{}".format(name), marker=dict(color='Grey', opacity=0.5)), row=(model + 1), col=column_2)
    fig.add_trace(
        go.Scatter(x=x, y=[0 for _ in range(len(x))], mode='lines', name='lines',
                   line=dict(color='grey', width=1, dash='dash')),
        row=(model + 1), col=column_2)


    fig.update_xaxes(title_text="Actual value", row=(model + 1), col=column_1,
                     title_font=dict(size=10), tickfont=dict(size=9))
    fig.update_yaxes(title_text="Predict value", row=(model + 1), col=column_1,
                     title_font=dict(size=10), tickfont=dict(size=9))

    fig.update_xaxes(title_text="Predict value", row=(model + 1), col=column_2,
                     title_font=dict(size=10), tickfont=dict(size=9))
    fig.update_yaxes(title_text="Residual value", row=(model + 1), col=column_2,
                     title_font=dict(size=10), tickfont=dict(size=9))


def h2h3_scatterplot_prediction_residual(dataset, x, fig, name, model, column_1, column_2):
    fig.add_trace(
        go.Scatter(x=dataset['actual_value'], y=dataset['{}_predict'.format(name)], mode="markers",
                   name="{}".format(name), marker=dict(color='Grey', opacity=0.5)), row=column_1, col=(model+1))
    fig.add_trace(
        go.Scatter(x=x, y=x, mode='lines', name='lines', line=dict(color='grey', width=1, dash='dash')),
        row=column_1, col=(model+1))

    fig.add_trace(
        go.Scatter(x=dataset['{}_predict'.format(name)], y=dataset['{}_residual'.format(name)],
                   mode="markers",
                   name="{}".format(name), marker=dict(color='Grey', opacity=0.5)), row=column_2, col=(model+1))
    fig.add_trace(
        go.Scatter(x=x, y=[0 for _ in range(len(x))], mode='lines', name='lines',
                   line=dict(color='grey', width=1, dash='dash')),
        row=column_2, col=(model+1))


    fig.update_xaxes(title_text="Actual value", row=column_1, col=(model+1),
                     title_font=dict(size=10), tickfont=dict(size=9))
    fig.update_yaxes(title_text="Predict value", row=column_1, col=(model+1),
                     title_font=dict(size=10), tickfont=dict(size=9))

    fig.update_xaxes(title_text="Predict value", row=column_2, col=(model+1),
                     title_font=dict(size=10), tickfont=dict(size=9))
    fig.update_yaxes(title_text="Residual value", row=column_2, col=(model+1),
                     title_font=dict(size=10), tickfont=dict(size=9))


def h4_scatterplot_prediction_residual(dataset, x, fig, name, col_num, row_1, row_2):
    fig.add_trace(
        go.Scatter(x=dataset['actual_value'], y=dataset['{}_predict'.format(name)], mode="markers",
                   name="{}".format(name), marker=dict(color='Grey', opacity=0.5)), row=row_1, col=col_num)
    fig.add_trace(
        go.Scatter(x=x, y=x, mode='lines', name='lines', line=dict(color='grey', width=1, dash='dash')),
        row=row_1, col=col_num)

    fig.add_trace(
        go.Scatter(x=dataset['{}_predict'.format(name)], y=dataset['{}_residual'.format(name)],
                   mode="markers",
                   name="{}".format(name), marker=dict(color='Grey', opacity=0.5)), row=row_2, col=col_num)
    fig.add_trace(
        go.Scatter(x=x, y=[0 for _ in range(len(x))], mode='lines', name='lines',
                   line=dict(color='grey', width=1, dash='dash')),
        row=row_2, col=col_num)

    fig.update_xaxes(title_text="Actual value", row=row_1, col=col_num,
                     title_font=dict(size=15), tickfont=dict(size=15))
    fig.update_yaxes(title_text="Predict value", row=row_1, col=col_num,
                     title_font=dict(size=15), tickfont=dict(size=15))

    fig.update_xaxes(title_text="Predict value", row=row_2, col=col_num,
                     title_font=dict(size=15), tickfont=dict(size=15))
    fig.update_yaxes(title_text="Residual value", row=row_2, col=col_num,
                     title_font=dict(size=15), tickfont=dict(size=15))


def h4_one_day_line_visualization(dataset, fig, month, day, row_num):
    target_dataset = dataset.loc[dataset['month'] == month]
    target_dataset = target_dataset.loc[target_dataset['day'] == day]
    target_dataset = target_dataset.reset_index(drop=True)

    target_datetime = target_dataset['tgt_datetime']
    actual_value_column = target_dataset['actual_value']

    model_column = target_dataset.iloc[:, 2:13]
    n_model_col = model_column.shape[1]


    model_colors = ['darkblue', 'darkviolet', 'lightcoral', 'limegreen', 'olive', 'orange', 'pink', 'red', 'sandybrown', 'steelblue', 'yellow']

    if month == 3:
        fig.add_trace(go.Scatter(x=target_datetime,
                                 y=actual_value_column,
                                 mode='lines+markers',
                                 line=dict(color='black', width=3),
                                 name=actual_value_column.name,
                                 ), row=row_num, col=1)

    else:
        fig.add_trace(go.Scatter(x=target_datetime,
                                 y=actual_value_column,
                                 mode='lines+markers',
                                 line=dict(color='black', width=3),
                                 name=actual_value_column.name,
                                 showlegend=False
                                 ), row=row_num, col=1)


    for i in range(0, n_model_col):
        if month == 3:
            fig.add_trace(go.Scatter(x=target_datetime,
                                     y=model_column.iloc[:, i],
                                     mode='lines+markers',
                                     line=dict(color=model_colors[i], width=1),
                                     name=model_column.columns.values[i],
                                     ), row=row_num, col=1)

        else:
            fig.add_trace(go.Scatter(x=target_datetime,
                                     y=model_column.iloc[:, i],
                                     mode='lines+markers',
                                     line=dict(color=model_colors[i], width=1),
                                     name=model_column.columns.values[i],
                                     showlegend=False
                                     ), row=row_num, col=1)

    fig.update_xaxes(title_text="Time", row=row_num, col=1,
                     title_font=dict(size=15), tickfont=dict(size=15))
    fig.update_yaxes(title_text="Power [kW]", row=row_num, col=1,
                     title_font=dict(size=15), tickfont=dict(size=15))


def h4_one_day_line_visualization_decem(dataset, fig, month, day, row_num):
    target_dataset = dataset.loc[dataset['month'] == month]
    target_dataset = target_dataset.loc[target_dataset['day'] == day]
    target_dataset = target_dataset.reset_index(drop=True)

    target_datetime = target_dataset['tgt_datetime']
    actual_value_column = target_dataset['actual_value']

    model_column = target_dataset.iloc[:, 2:13]
    n_model_col = model_column.shape[1]

    model_colors = ['darkblue', 'darkviolet', 'lightcoral', 'limegreen', 'olive', 'orange', 'pink', 'red', 'sandybrown', 'steelblue', 'yellow']

    if month == 12:
        fig.add_trace(go.Scatter(x=target_datetime,
                                 y=actual_value_column,
                                 mode='lines+markers',
                                 line=dict(color='black', width=3),
                                 name=actual_value_column.name,
                                 ), row=row_num, col=1)

    else:
        fig.add_trace(go.Scatter(x=target_datetime,
                                 y=actual_value_column,
                                 mode='lines+markers',
                                 line=dict(color='black', width=3),
                                 name=actual_value_column.name,
                                 showlegend=False
                                 ), row=row_num, col=1)


    for i in range(0, n_model_col):
        if month == 12:
            fig.add_trace(go.Scatter(x=target_datetime,
                                     y=model_column.iloc[:, i],
                                     mode='lines+markers',
                                     line=dict(color=px.colors.qualitative.Safe[i], width=1),
                                     name=model_column.columns.values[i],
                                     ), row=row_num, col=1)

        else:
            fig.add_trace(go.Scatter(x=target_datetime,
                                     y=model_column.iloc[:, i],
                                     mode='lines+markers',
                                     line=dict(color=px.colors.qualitative.Safe[i], width=1),
                                     name=model_column.columns.values[i],
                                     showlegend=False
                                     ), row=row_num, col=1)

    fig.update_xaxes(title_text="Time", row=row_num, col=1,
                     title_font=dict(size=15), tickfont=dict(size=15))
    fig.update_yaxes(title_text="Power [kW]", row=row_num, col=1,
                     title_font=dict(size=15), tickfont=dict(size=15))


class DlVisualization(DnnModel):
    def __init__(self, scaler_x, scaler_y, train_x, train_y, test_x, test_y, test_set_time_dummy, model):
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.test_set_time_dummy = test_set_time_dummy
        self.model = model

    def dl_predict(self, path, his, title):
        yhat_dl = self.model.predict(self.test_x, verbose=1)
        predicting_process(self.scaler_y, self.test_set_time_dummy, yhat_dl, self.test_y, path, his, title)


class MlVisualization(MlModel):
    def __init__(self, scaler_x, scaler_y, train_x, train_y, test_x, test_y, test_set_time_dummy, model):
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.test_set_time_dummy = test_set_time_dummy
        self.model = model

    def ml_predict(self, path, his, title):
        yhat_ml = self.model.predict(self.test_x)
        predicting_process(self.scaler_y, self.test_set_time_dummy, yhat_ml, self.test_y, path, his, title)


    def ml_predict_tree_based(self, path, his, title):
        yhat_ml = self.model.predict(self.test_x)
        predicting_process_tree_based(self.test_set_time_dummy, yhat_ml, self.test_y, path, his, title)

class PredictVisualization:
    def __init__(self, dataset):
        self.dataset = dataset

    # def scatter_matrix_visualization(self, plant, his):
    #     target_column = self.dataset.iloc[:, 1:]
    #     fig = px.scatter_matrix(target_column,
    #                             title=f'{plant}_h{his}_scattermatrix',
    #                             labels={col: col.split('_')[0] for col in target_column.columns.values})
    #     fig.update_traces(diagonal_visible=False)
    #     po.plot(fig, filename=f'./predictionresult/scattermatrix/{plant}_h{his}_scattermatrix.html')

    def line_visualization(self, plant, his):
        target_datetime = self.dataset['tgt_datetime']
        target_column = self.dataset.iloc[:, 1:]

        n_col = target_column.shape[1]

        colors = ['olivedrab', 'darkblue', 'darkviolet', 'lightcoral', 'limegreen', 'olive', 'orange', 'pink', 'red', 'sandybrown', 'steelblue', 'yellow']

        fig = go.Figure()

        for i in range(0, n_col):
            fig.add_trace(go.Scatter(x=target_datetime,
                                     y=target_column.iloc[:, i],
                                     mode='lines+markers',
                                     line=dict(color=colors[i]),
                                     name=target_column.columns.values[i]))

        fig.update_layout(title=f'{plant}_h{his}_line graph')
        po.plot(fig, filename=f'./predictionresult/linegraph/{plant}_h{his}_line graph.html')


    def scatterplot_matrix_actual_and_prediction(self, fig, plant, his, model_name):
        if plant == 'hk1':
            x = list(range(-250, 1800))
        elif plant == 'hk2':
            x = list(range(-500, 3400))
        else:
            x = list(range(-150, 2300))

        for i, name in enumerate(model_name):
            fig.add_trace(
                go.Scatter(x=self.dataset['actual_value'], y=self.dataset['{}_predict'.format(name)], mode="markers",
                           name="{}".format(name), marker=dict(color='Grey', opacity=0.5)), row=(i + 1),
                col=(his + 1))
            fig.add_trace(
                go.Scatter(x=x, y=x, mode='lines', name='lines', line=dict(color='grey', width=1, dash='dash')),
                row=(i + 1), col=(his + 1))
            fig.update_xaxes(title_text="Actual value", row=(i + 1), col=(his + 1),
                             title_font=dict(size=10), tickfont=dict(size=9))
            fig.update_yaxes(title_text="Predict value", row=(i + 1), col=(his + 1),
                             title_font=dict(size=10), tickfont=dict(size=9))

    def h2h3_scatterplot_matrix_prediction_and_residual(self, fig, plant, model_name):
        if plant == 'hk1':
            x = list(range(-500, 2000))
        elif plant == 'hk2':
            x = list(range(-500, 3500))
        else:
            x = list(range(-500, 2500))

        for i, name in enumerate(model_name):
            h2h3_scatterplot_prediction_residual(self.dataset, x, fig, name, i, 1, 2)

    def h2h3_scatterplot_matrix_total_plants_prediction(self, fig, plant, model_name):
        for i, name in enumerate(model_name):
            if plant == 'hk1':
                x = list(range(-500, 2000))
                scatterplot_prediction_residual_common(self.dataset, x, fig, name, i, 1, 2)
            elif plant == 'hk2':
                x = list(range(-500, 3500))
                scatterplot_prediction_residual_common(self.dataset, x, fig, name, i, 3, 4)
            else:
                x = list(range(-500, 2500))
                scatterplot_prediction_residual_common(self.dataset, x, fig, name, i, 5, 6)


    def h4_scatterplot_matrix_total_plants_prediction(self, fig, plant, model_name):
        if plant == 'hk1':
            x = list(range(-500, 2000))
            h4_scatterplot_prediction_residual(self.dataset, x, fig, model_name, 1, 1, 2)
        elif plant == 'hk2':
            x = list(range(-500, 3500))
            h4_scatterplot_prediction_residual(self.dataset, x, fig, model_name, 2, 1, 2)
        else:
            x = list(range(-500, 2500))
            h4_scatterplot_prediction_residual(self.dataset, x, fig, model_name, 3, 1, 2)


    def monthly_line_visualization(self, fig, his, month):
        target_dataset = self.dataset.loc[self.dataset['month'] == month]
        target_dataset = target_dataset.reset_index(drop=True)

        target_datetime = target_dataset['tgt_datetime']
        target_column = target_dataset.iloc[:, 1:13]
        n_col = target_column.shape[1]

        colors = ['olivedrab', 'darkblue', 'darkviolet', 'lightcoral', 'limegreen', 'olive', 'orange', 'pink',
                  'red', 'sandybrown', 'steelblue', 'yellow']

        for i in range(0, n_col):
            # if his == 0:
            fig.add_trace(go.Scatter(x=target_datetime,
                                     y=target_column.iloc[:, i],
                                     mode='lines+markers',
                                     line=dict(color=colors[i], width=1),
                                     name=target_column.columns.values[i],
                                     ), row=his+1, col=1)

            # else:
            #     fig.add_trace(go.Scatter(x=target_datetime,
            #                              y=target_column.iloc[:, i],
            #                              mode='lines+markers',
            #                              line=dict(color=colors[i], width=1),
            #                              name=target_column.columns.values[i],
            #                              showlegend=False), row=his + 1, col=1)



    def h2h3_yearly_line_visualization(self, fig):
        target_datetime = self.dataset['tgt_datetime']

        actual_value_column = self.dataset['actual_value']
        single_type_column = self.dataset.iloc[:, 2:8]
        ensemble_type_column = self.dataset.iloc[:, 8:13]

        target_single_column = pd.concat([actual_value_column, single_type_column], axis=1)
        target_ensemble_column = pd.concat([actual_value_column, ensemble_type_column], axis=1)

        n_single_col = target_single_column.shape[1]
        n_ensemble_col = target_ensemble_column.shape[1]

        colors_single = ['olivedrab', 'darkblue', 'darkviolet', 'lightcoral', 'limegreen', 'olive', 'orange']
        colors_ensemble = ['olivedrab', 'pink', 'red', 'sandybrown', 'steelblue', 'yellow']

        for i in range(0, n_single_col):
            fig.add_trace(go.Scatter(x=target_datetime,
                                     y=target_single_column.iloc[:, i],
                                     mode='lines+markers',
                                     line=dict(color=colors_single[i], width=1),
                                     name=target_single_column.columns.values[i],
                                     ), row=1, col=1)

        for j in range(0, n_ensemble_col):
            fig.add_trace(go.Scatter(x=target_datetime,
                                     y=target_ensemble_column.iloc[:, j],
                                     mode='lines+markers',
                                     line=dict(color=colors_ensemble[j], width=1),
                                     name=target_ensemble_column.columns.values[j],
                                     ), row=2, col=1)



    def h2h3_monthly_line_visualization(self, fig, month):
        target_dataset = self.dataset.loc[self.dataset['month'] == month]
        target_dataset = target_dataset.reset_index(drop=True)

        target_datetime = self.dataset['tgt_datetime']

        actual_value_column = self.dataset['actual_value']
        single_type_column = self.dataset.iloc[:, 2:8]
        ensemble_type_column = self.dataset.iloc[:, 8:13]

        target_single_column = pd.concat([actual_value_column, single_type_column], axis=1)
        target_ensemble_column = pd.concat([actual_value_column, ensemble_type_column], axis=1)

        n_single_col = target_single_column.shape[1]
        n_ensemble_col = target_ensemble_column.shape[1]

        colors_single = ['olivedrab', 'darkblue', 'darkviolet', 'lightcoral', 'limegreen', 'olive', 'orange']
        colors_ensemble = ['olivedrab', 'pink', 'red', 'sandybrown', 'steelblue', 'yellow']

        for i in range(0, n_single_col):
            fig.add_trace(go.Scatter(x=target_datetime,
                                     y=target_single_column.iloc[:, i],
                                     mode='lines+markers',
                                     line=dict(color=colors_single[i], width=1),
                                     name=target_single_column.columns.values[i],
                                     ), row=1, col=1)

        for j in range(0, n_ensemble_col):
            fig.add_trace(go.Scatter(x=target_datetime,
                                     y=target_ensemble_column.iloc[:, j],
                                     mode='lines+markers',
                                     line=dict(color=colors_ensemble[j], width=1),
                                     name=target_ensemble_column.columns.values[j],
                                     ), row=2, col=1)


    def h4_one_day_lineplot(self, fig, month_list, day):
        for i, month in enumerate(month_list):
            if month == 3:
                h4_one_day_line_visualization(self.dataset, fig, month, day, 1)
            elif month == 6:
                h4_one_day_line_visualization(self.dataset, fig, month, day, 2)
            elif month == 9:
                h4_one_day_line_visualization(self.dataset, fig, month, day, 3)
            else:
                h4_one_day_line_visualization(self.dataset, fig, month, day, 4)


    def h4_one_day_lineplot_dec(self, fig, month, day):
        h4_one_day_line_visualization_decem(self.dataset, fig, month, day, 1)