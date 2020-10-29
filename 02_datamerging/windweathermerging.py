from dataload.load import WindWeather

import pandas as pd

class Merging(WindWeather):
    def __init__(self):
        super().__init__()

    def data_merge(self, labeling):
        print("----------------------------------------------------------------------------")
        print("Wind Generation")
        print(self.wind_dataset.isnull().sum())
        print("----------------------------------------------------------------------------")
        print("Weather Forecast")
        print(self.weather_dataset.isnull().sum())
        print("----------------------------------------------------------------------------")
        self.weather_dataset = self.weather_dataset.drop(['anc_time'], axis=1)
        self.weather_dataset = self.weather_dataset.drop(['tgt_beg'], axis=1)
        self.weather_dataset = self.weather_dataset[
            ['tgt_Y', 'tgt_M', 'tgt_D', 'tgt_tz', 'temp3h', 'humidity', 'windspeed', 'winddirection',
             'rain6h', 'snow6h', 'rainprobability', 'raintype', 'seawave', 'skystatus']]

        # wind_data_path_hk = wind_data_path_hk.drop(['group'], axis=1)
        self.wind_dataset = self.wind_dataset.drop(['tgt_time'], axis=1)
        merged_dataset = pd.merge(self.wind_dataset, self.weather_dataset, how='inner', on=['tgt_Y', 'tgt_M', 'tgt_D', 'tgt_tz'])
        merged_dataset = merged_dataset[
            ['tgt_Y', 'tgt_M', 'tgt_D', 'tgt_tz', 'temp3h', 'humidity', 'windspeed', 'winddirection',
             'rain6h', 'snow6h', 'rainprobability', 'raintype', 'seawave', 'skystatus', labeling]]

        print("----------------------------------------------------------------------------")
        print(merged_dataset.info())
        print("Total Data Set matrix : ", merged_dataset.shape)
        merged_dataset.to_csv('./dataset/mergeddata/wind_weather_forecast_{}_4hours.csv'.format(labeling), index=False)
        print("----------------------------------------------------------------------------")