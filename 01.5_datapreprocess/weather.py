from dataload.load import Weather
import datetime

import pandas as pd


class WeatherDataset(Weather):
    def __init__(self):
        super().__init__()

    def data_filtering(self, labeling):
        self.dataset = self.dataset.rename(columns={self.dataset.columns[0]: "day"})
        self.dataset = self.dataset.rename(columns={self.dataset.columns[3]: labeling})
        # 4시간 데이터 선택
        self.dataset = self.dataset[(self.dataset['forecast'] == 4)]
        # 25시간 데이터 선택
        # weather_for_data = weather_for_data[(weather_for_data['forecast'] == 25)]
        self.dataset['hour'] = self.dataset['hour'].apply(lambda x: int(int(x) / 100))
        self.dataset['forecast'] = self.dataset['forecast'].apply(lambda x: int(x))
        print("----------------------------------------------------------------------------")
        print(self.dataset.info())
        print("----------------------------------------------------------------------------")
        print(self.dataset.isnull().sum())
        print("----------------------------------------------------------------------------")

    def data_datetime_setting(self, year=2014):
        self.dataset['anc_time'] = pd.date_range(start='{}-01-01 02:00'.format(year), periods=self.dataset.shape[0],
                                                 freq='3H')
        self.dataset['tgt_beg'] = self.dataset['anc_time'].apply(lambda x: x + datetime.timedelta(hours=4))
        self.dataset = self.dataset.drop(['day', 'hour', 'forecast'], axis=1)
        return self.dataset


class WeatherMerging:
    def __init__(self):
        pass

    def data_merging(self, temp_dataset, location='hankyeong', year=2014):
        merged_dataset = temp_dataset[0]

        for i in range(1, len(temp_dataset)):
            merged_dataset = pd.merge(merged_dataset, temp_dataset[i], on=['anc_time', 'tgt_beg'], how='outer')

        merged_dataset['anc_time'] = pd.to_datetime(merged_dataset['anc_time'])
        merged_dataset['tgt_beg'] = pd.to_datetime(merged_dataset['tgt_beg'])
        merged_dataset['tgt_Y'] = merged_dataset['tgt_beg'].apply(lambda x: x.year)
        merged_dataset['tgt_M'] = merged_dataset['tgt_beg'].apply(lambda x: x.month)
        merged_dataset['tgt_D'] = merged_dataset['tgt_beg'].apply(lambda x: x.day)
        merged_dataset['tgt_tz'] = merged_dataset['tgt_beg'].apply(lambda x: x.hour / 3)

        print(merged_dataset.info())
        print("----------------------------------------------------------------------------")
        print("Save matrix:", merged_dataset.shape)
        merged_dataset.to_csv(f'./dataset/csvdata/weather_dataset/{location}/{year}/weather_forecast_{year}_4hours_target.csv',
                              index=False)
        print("----------------------------------------------------------------------------")

    def dataset_merging_deleting_null_value(self, weather_2014_path, weather_2015_path, weather_2016_path,
                                            weather_2017_path, labeling):
        weather_2014 = pd.read_csv(weather_2014_path)
        weather_2015 = pd.read_csv(weather_2015_path)
        weather_2016 = pd.read_csv(weather_2016_path)
        weather_2017 = pd.read_csv(weather_2017_path)
        weather_forecast_dataset = pd.concat([weather_2014, weather_2015, weather_2016, weather_2017], axis=0)
        weather_forecast_dataset.isnull().sum()
        print("----------------------------------------------------------------------------")
        print("Save matrix:", weather_forecast_dataset.shape)
        print(weather_forecast_dataset.info())
        print("----------------------------------------------------------------------------")
        weather_forecast_dataset.to_csv('./dataset/csvdata/weather_dataset/weather_forecast_{}_4hours.csv'.format(labeling), index=False)
