from dataload.load import Wind

import pandas as pd
import numpy as np
import datetime


# Original .xlsx file load
class WindRawDataset(Wind):
    def __init__(self):
        super().__init__()

    # Null value fill 0 and save as .csv file
    def dataset_to_csv(self, title):
        self.dataset = self.dataset.fillna(0)
        self.dataset.to_csv('./dataset/csvdata/wind_raw_dataset/{}_raw_data.csv'.format(title), index=False)


# Converted .csv file load and preprocessing
class WindDataset:
    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        self.dummy_dataset = None
        self.generation_dataset = None

    def time_split(self):
        # Datetime setting
        self.dataset = self.dataset.rename(columns={"PCTimeStamp": "datetime"})

        # Dummy dataset setting
        self.dummy_dataset = pd.DataFrame(columns=['datetime', 'Y', 'M', 'D'])

        # Input datetime into dummy dataset
        self.dummy_dataset['datetime'] = self.dataset['datetime'].apply(lambda x: str(x))
        self.dummy_dataset['Y_M_D'] = self.dummy_dataset['datetime'].apply(lambda x: x.split(' ')[0])
        self.dummy_dataset['Y'] = self.dummy_dataset['Y_M_D'].apply(lambda x: x.split('-')[0])
        self.dummy_dataset['M'] = self.dummy_dataset['Y_M_D'].apply(lambda x: x.split('-')[1])
        self.dummy_dataset['D'] = self.dummy_dataset['Y_M_D'].apply(lambda x: x.split('-')[2])

        # Base datetime value generate : ex)2014-01-01 00:00:00
        datetime_base = datetime.datetime(int(self.dummy_dataset['Y'][0]), int(self.dummy_dataset['M'][0]),
                                          int(self.dummy_dataset['D'][0]), 0, 0, 0)

        # Datetime generating every 10 minutes
        self.dummy_dataset['Time'] = ''
        self.dummy_dataset['Time'][0] = datetime_base
        self.dummy_dataset['Time'] = pd.date_range(self.dummy_dataset['Time'][0], periods=210384, freq='10min')
        self.dummy_dataset['H'] = self.dummy_dataset['Time'].apply(lambda x: x.hour)
        self.dummy_dataset['min'] = self.dummy_dataset['Time'].apply(lambda x: x.minute)

    def generation_max_select(self):
        self.dataset = self.dataset.drop(['datetime'], axis=1)
        self.dataset['generation'] = self.dataset.max(axis=1)
        self.dataset['generation'] = self.dataset['generation'].apply(lambda x: 0 if x < 0 else x)
        self.generation_dataset = self.dataset['generation']

    def generation_max_select_transformed_kw(self):
        self.dataset = self.dataset.drop(['datetime'], axis=1)
        self.dataset['generation'] = self.dataset.max(axis=1)
        self.dataset['generation'] = self.dataset['generation'].apply(lambda x: 0 if x < 0 else x)
        self.dataset['generation'] = self.dataset['generation'].apply(lambda x: ((x * 6) / 1000))
        self.generation_dataset = self.dataset['generation']

    def dummy_dataset_and_generation_dataset_merge(self, labeling):
        self.dummy_dataset[labeling] = self.generation_dataset
        self.dummy_dataset = self.dummy_dataset.drop(['datetime', 'Y_M_D', 'Time'], axis=1)
        self.dummy_dataset = self.dummy_dataset.drop(self.dummy_dataset.index[0:28])
        self.dummy_dataset = self.dummy_dataset.reset_index(drop=True)

        GROUPSIZE = 18
        n_row = self.dummy_dataset.shape[0]
        n_group = np.ceil(n_row / GROUPSIZE)
        n_rest = n_row % GROUPSIZE

        group_key = np.arange(n_group, dtype=np.int)
        group_column = np.repeat(group_key, GROUPSIZE)

        self.dummy_dataset['group'] = group_column[:-(GROUPSIZE - n_rest)]
        self.dummy_dataset = self.dummy_dataset.groupby(['group']).mean()

        self.dummy_dataset['tgt_time'] = None

        self.dummy_dataset['tgt_time'] = pd.date_range(start='2014-01-01 06:00',
                                                       periods=self.dummy_dataset.shape[0], freq='3H')
        self.dummy_dataset = self.dummy_dataset.drop(['H', 'min'], axis=1)

        self.dummy_dataset['tgt_Y'] = self.dummy_dataset['tgt_time'].apply(lambda x: x.year)
        self.dummy_dataset['tgt_M'] = self.dummy_dataset['tgt_time'].apply(lambda x: x.month)
        self.dummy_dataset['tgt_D'] = self.dummy_dataset['tgt_time'].apply(lambda x: x.day)
        self.dummy_dataset['tgt_tz'] = self.dummy_dataset['tgt_time'].apply(lambda x: x.hour / 3)

        self.dummy_dataset = self.dummy_dataset[['tgt_time', 'tgt_Y', 'tgt_M', 'tgt_D', 'tgt_tz', labeling]]

        self.dummy_dataset.to_csv('./dataset/csvdata/wind_dataset/wind_generation_{}_dataset.csv'.format(labeling),
                                  index=False)

        print("----------------------------------------------------------------------------")
        print(self.dummy_dataset.info())
        print("Save matrix:", self.dummy_dataset.shape)
        print("----------------------------------------------------------------------------")





