from datamerging.predictionresultmerging import DatasetMerging
from datavisualization.visualization import PredictVisualization
from plotly.subplots import make_subplots
import plotly.offline as po
import pandas as pd

power_plants = ['hk1', 'hk2', 'ss']
models_name = ['PLS', 'ElasticNet', 'DT', 'SVR', 'MLP', 'RF', 'Bag', 'Ada', 'GBR', 'XGB', 'LGBM']
total_models_name = ['MLR', 'PLS', 'ElasticNet', 'DT', 'SVR', 'MLP', 'RF', 'Bag', 'Ada', 'GBR', 'XGB', 'LGBM']
column_title = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']
h2h3_row_title = ['Actual-Predict', 'Predict-Residual']
h2h3_column_title = ['MLR Actual-Predict', 'PLS Actual-Predict', 'ElasticNet Actual-Predict', 'DT Actual-Predict',
                     'SVR Actual-Predict', 'MLP Actual-Predict', 'RF Actual-Predict', 'BAG Actual-Predict',
                     'ADA Actual-Predict', 'GBR Actual-Predict', 'XGB Actual-Predict',
                     'MLR Predict-Residual', 'PLS Predict-Residual', 'ElasticNet Predict-Residual', 'DT Predict-Residual',
                     'SVR Predict-Residual', 'MLP Predict-Residual', 'RF Predict-Residual', 'BAG Predict-Residual',
                     'ADA Predict-Residual', 'GBR Predict-Residual', 'XGB Predict-Residual']
h2h3_total_power_column_title = ['HK1 Actual-Predict', 'HK1 Predict-Residual', 'HK2 Actual-Predict',
                                 'HK2 Predict-Residual', 'SS Actual-Predict', 'SS Predict-Residual']
month_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
model_type = ['Single model', 'Ensemble model']
column_title_separate = ['MLR h0', 'MLR h1', 'MLR h2', 'MLR h3', 'MLR h4', 'MLR h5',
                         'PLS h0', 'PLS h1', 'PLS h2', 'PLS h3', 'PLS h4', 'PLS h5',
                         'ElasticNet h0', 'ElasticNet h1', 'ElasticNet h2', 'ElasticNet h3', 'ElasticNet h4', 'ElasticNet h5',
                         'DT h0', 'DT h1', 'DT h2', 'DT h3', 'DT h4', 'DT h5',
                         'SVR h0', 'SVR h1', 'SVR h2', 'SVR h3', 'SVR h4', 'SVR h5',
                         'MLP h0', 'MLP h1', 'MLP h2', 'MLP h3', 'MLP h4', 'MLP h5',
                         'RF h0', 'RF h1', 'RF h2', 'RF h3', 'RF h4', 'RF h5',
                         'BAG h0', 'BAG h1', 'BAG h2', 'BAG h3', 'BAG h4', 'BAG h5',
                         'ADA h0', 'ADA h1', 'ADA h2', 'ADA h3', 'ADA h4', 'ADA h5',
                         'GBR h0', 'GBR h1', 'GBR h2', 'GBR h3', 'GBR h4', 'GBR h5',
                         'XGB h0', 'XGB h1', 'XGB h2', 'XGB h3', 'XGB h4', 'XGB h5']

h4_total_models_name = 'XGB'
h4_row_title = ['Actual-Predict', 'Predict-Residual']
h4_column_title = ['HK1 XGB', 'HK2 XGB', 'SS XGB']
h4_subplot_title = ['HK1 XGB', 'HK2 XGB', 'SS XGB', 'HK1 XGB', 'HK2 XGB', 'SS XGB']

h4_line_month_list = [3, 6, 9, 12]


def dataset_merging_process():
    for plant in power_plants:
        total_result = None
        for his in range(6):
            lr_dataset = DatasetMerging().data_load_lr(plant, his)

            for model in models_name:
                dataset_model = DatasetMerging().data_load(plant, his, model)

                if model == 'PLS':
                    total_result = DatasetMerging().data_merging_initial(lr_dataset, dataset_model, model)
                else:
                    total_result = DatasetMerging().data_merging(total_result, dataset_model, model)

            for model in total_models_name:
                total_result = DatasetMerging().data_residual(total_result, model)

            DatasetMerging().data_saving(total_result, plant, his)

# def dataset_scatterplot():
#     target_dataset = DatasetMerging().total_dataset_load('hk1', 0)
#     PredictVisualization(target_dataset).scatterplot_actual_and_prediction('PLS_predict')
#
# def dataset_scattermatrix():
#     for plant in power_plants:
#         for his in range(6):
#             target_dataset = DatasetMerging().total_dataset_load(plant, his)
#             PredictVisualization(target_dataset).scatter_matrix_visualization(plant, his)
#
# def dataset_lineplot():
#     for plant in power_plants:
#         for his in range(6):
#             target_dataset = DatasetMerging().total_dataset_load(plant, his)
#             PredictVisualization(target_dataset).line_visualization(plant, his)

def total_scatterplot_matrix():
    for plant in power_plants:
        fig = make_subplots(rows=11, cols=6,  vertical_spacing=0.03, row_titles=total_models_name,
                            subplot_titles=column_title_separate, horizontal_spacing=0.05)
        for his in range(6):
            target_dataset = DatasetMerging().total_dataset_load(plant, his)
            PredictVisualization(target_dataset).scatterplot_matrix_actual_and_prediction(fig, plant, his, total_models_name)
        fig.update_layout(showlegend=False, title_text="{}".format(plant), font=dict(size=18), height=3800, width=2100)
        po.plot(fig, filename=f'./predictionresult/scattermatrix/{plant}_scatterplot matrix.html', auto_open=False)
        fig.write_image(f'./predictionresult/scattermatrix/{plant}_scatterplot matrix.pdf')

def h2h3_scatterplot_matrix():
    for his in range(2, 4):
        for plant in power_plants:
            fig = make_subplots(rows=2, cols=11, column_widths=[0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
                                subplot_titles=h2h3_column_title)
            target_dataset = DatasetMerging().total_dataset_load(plant, his)
            PredictVisualization(target_dataset).h2h3_scatterplot_matrix_prediction_and_residual(fig, plant, total_models_name)
            fig.update_layout(showlegend=False, title_text=f"{plant}_h{his}", font=dict(size=18), height=1000, width=4500)
            po.plot(fig, filename=f'./predictionresult/scattermatrix/{plant}_{his}_scatterplot matrix.html', auto_open=False)
            fig.write_image(f'./predictionresult/scattermatrix/{plant}_{his}_scatterplot matrix.pdf')

def h2h3_total_plants_scatterplot_matrix():
    for his in range(2, 4):
        fig = make_subplots(rows=11, cols=6, vertical_spacing=0.03, row_titles=total_models_name,
                            column_titles=h2h3_total_power_column_title)
        for plant in power_plants:
            target_dataset = DatasetMerging().total_dataset_load(plant, his)
            PredictVisualization(target_dataset).h2h3_scatterplot_matrix_total_plants_prediction(fig, plant, total_models_name)

        fig.update_layout(showlegend=False, title_text=f"h{his}", font=dict(size=18), height=5000, width=2000, autosize=False)
        po.plot(fig, filename=f'./predictionresult/scattermatrix/{his}_scatterplot matrix.html', auto_open=False)
        fig.write_image(f'./predictionresult/scattermatrix/{his}_scatterplot matrix.png')

def h4_total_plants_scatterplot_matrix():
    his = 4
    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.15, horizontal_spacing=0.1, subplot_titles=h4_subplot_title)
    for plant in power_plants:
        target_dataset = DatasetMerging().total_dataset_load(plant, his)
        PredictVisualization(target_dataset).h4_scatterplot_matrix_total_plants_prediction(fig, plant,
                                                                                             h4_total_models_name)

    fig.update_layout(showlegend=False, title_text=f"h{his}", font=dict(size=25), height=1000, width=1500,
                      autosize=False)
    po.plot(fig, filename=f'./predictionresult/scattermatrix/{his}_scatterplot matrix.html', auto_open=False)
    fig.write_image(f'./predictionresult/scattermatrix/{his}_scatterplot matrix.png')
    fig.write_image(f'./predictionresult/scattermatrix/{his}_scatterplot matrix.svg')

def monthly_lineplot():
    for plant in power_plants:
        for month in month_list:
            fig = make_subplots(rows=6, cols=1, vertical_spacing=0.05, row_titles=column_title,
                                row_heights=[0.17, 0.17, 0.17, 0.17, 0.17, 0.17])
            for his in range(6):
                target_dataset = DatasetMerging().total_dataset_load(plant, his)
                target_dataset['month'] = pd.DatetimeIndex(target_dataset['tgt_datetime']).month
                PredictVisualization(target_dataset).monthly_line_visualization(fig, his, month)

            fig.update_layout(showlegend=True, title_text=f"{plant}_{month}", font=dict(size=18), width=2500,
                              height=1800)
            po.plot(fig,
                    filename=f'./predictionresult/linegraph/{plant}/{month}/{plant}_{month}_time_series_graph.html',
                    auto_open=False)
            fig.write_image(f'./predictionresult/linegraph/{plant}/{month}/{plant}_{month}_time_series_graph.png')

def h2h3_result_of_lineplot():
    for plant in power_plants:
        # 1-year version
        for his in range(2, 4):
            fig_1 = make_subplots(rows=2, cols=1, vertical_spacing=0.05, row_titles=model_type, row_heights=[0.5, 0.5])
            target_dataset = DatasetMerging().total_dataset_load(plant, his)
            PredictVisualization(target_dataset).h2h3_yearly_line_visualization(fig_1)

            fig_1.update_layout(showlegend=True, title_text=f"{plant}_h{his}", font=dict(size=18), width=2000,
                              height=1500)
            po.plot(fig_1,
                    filename=f'./predictionresult/linegraph/{plant}/line_h{his}/{plant}_h{his}_time_series_graph.html',
                    auto_open=False)
            fig_1.write_image(f'./predictionresult/linegraph/{plant}/line_h{his}/{plant}_h{his}_time_series_graph.png')

        # monthly version
        for month in month_list:
            for his in range(2, 4):
                fig_2 = make_subplots(rows=2, cols=1, vertical_spacing=0.05, row_titles=model_type,
                                      row_heights=[0.5, 0.5])
                target_dataset = DatasetMerging().total_dataset_load(plant, his)
                target_dataset['month'] = pd.DatetimeIndex(target_dataset['tgt_datetime']).month
                PredictVisualization(target_dataset).h2h3_monthly_line_visualization(fig_2, his, month)

                fig_2.update_layout(showlegend=True, title_text=f"{plant}_{month}", font=dict(size=18), width=2500,
                                    height=1500)
                po.plot(fig_2,
                        filename=f'./predictionresult/linegraph/{plant}/line_h{his}/{month}/{plant}_h{his}_{month}_time_series_graph.html',
                        auto_open=False)
                fig_2.write_image(
                    f'./predictionresult/linegraph/{plant}/line_h{his}/{month}/{plant}_h{his}_{month}_time_series_graph.png')

def h4_result_of_lineplot():
    his = 4
    day = 1

    for plant in power_plants:
        fig_2 = make_subplots(rows=4, cols=1, vertical_spacing=0.1)
        target_dataset = DatasetMerging().total_dataset_load(plant, his)
        target_dataset['month'] = pd.DatetimeIndex(target_dataset['tgt_datetime']).month
        target_dataset['day'] = pd.DatetimeIndex(target_dataset['tgt_datetime']).day


        accum_target_dataset = pd.DataFrame()

        for i in h4_line_month_list:
                target_dataset_temp = target_dataset.loc[target_dataset['month'] == i]
                target_dataset_temp = target_dataset_temp.loc[target_dataset_temp['day'] == day]
                target_dataset_temp = target_dataset_temp.reset_index(drop=True)
                print(target_dataset_temp.head())

                accum_target_dataset = accum_target_dataset.append(target_dataset_temp, ignore_index=True)

        accum_target_dataset.to_csv(f'./3_6_9_12_dataset_{plant}.csv', index=False)

        PredictVisualization(target_dataset).h4_one_day_lineplot(fig_2, h4_line_month_list, day)

        fig_2.update_layout(legend_title_text='Model', font=dict(size=18), width=2000, height=1000)
        po.plot(fig_2,
                filename=f'./predictionresult/linegraph/{plant}/line_h{his}/{plant}_h{his}_time_series_graph_4days.html',
                auto_open=False)
        fig_2.write_image(
            f'./predictionresult/linegraph/{plant}/line_h{his}/{plant}_h{his}_time_series_graph_4days.png')


def h4_result_of_lineplot_decem():
    his = 4
    day = 1

    for plant in power_plants:
        fig_2 = make_subplots(rows=1, cols=1)
        target_dataset = DatasetMerging().total_dataset_load(plant, his)
        target_dataset['month'] = pd.DatetimeIndex(target_dataset['tgt_datetime']).month
        target_dataset['day'] = pd.DatetimeIndex(target_dataset['tgt_datetime']).day

        PredictVisualization(target_dataset).h4_one_day_lineplot_dec(fig_2, 12, day)

        fig_2.update_layout(legend_title_text='Model', font=dict(size=18), width=2000, height=1000)
        po.plot(fig_2,
                filename=f'./predictionresult/linegraph/{plant}/line_h{his}/{plant}_h{his}_time_series_graph_4days_dece.html',
                auto_open=False)
        fig_2.write_image(
            f'./predictionresult/linegraph/{plant}/line_h{his}/{plant}_h{his}_time_series_graph_4days_dec.png')


if __name__ == '__main__':
    # dataset_merging_process()
    # dataset_scatterplot()
    # dataset_scattermatrix()
    # dataset_lineplot()
    # total_scatterplot_matrix()
    # h2h3_scatterplot_matrix()
    # h2h3_total_plants_scatterplot_matrix()
    # monthly_lineplot()
    # h2h3_result_of_lineplot()

    h4_total_plants_scatterplot_matrix()
    # h4_result_of_lineplot()
    # h4_result_of_lineplot_decem()

    print('finish')
