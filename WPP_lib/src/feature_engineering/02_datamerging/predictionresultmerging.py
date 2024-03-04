import pandas as pd

class DatasetMerging:
    def __init__(self):
        self.lr_result = None

    def data_load_lr(self, plant, his):
        self.lr_result = pd.read_csv(f'./predictionresult/{plant}/h{his}/{plant} Generation_MLR H{his}.csv')
        self.lr_result = self.lr_result.rename(columns={'predict_value': 'MLR_predict'})
        self.lr_result = self.lr_result[['tgt_datetime', 'actual_value', 'MLR_predict']]
        return self.lr_result


    def data_load(self, plant, his, model):
        dataset = pd.read_csv(f'./predictionresult/{plant}/h{his}/{plant} Generation_{model} H{his}.csv')
        dataset = dataset.rename(columns={'predict_value': f'{model}_predict'})
        return dataset

    def data_merging_initial(self, lr_result, dataset, model):
        total_result = pd.concat([lr_result, dataset[f'{model}_predict']], axis=1)
        return total_result

    def data_merging(self, total_result, dataset, model):
        total_result = pd.concat([total_result, dataset[f'{model}_predict']], axis=1)
        return total_result

    def data_residual(self, total_result, model):
        total_result[f'{model}_residual'] = total_result[f'{model}_predict']-total_result['actual_value']
        return total_result

    def data_saving(self, total_result, plant, his):
        total_result.to_csv(f'./predictionresult/{plant} Generation_total_result_h{his}.csv', index=False)

    def total_dataset_load(self, plant, his):
        dataset = pd.read_csv(f'./predictionresult/{plant} Generation_total_result_h{his}.csv')
        return dataset