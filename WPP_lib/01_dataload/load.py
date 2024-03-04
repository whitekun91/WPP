import pandas as pd

# Original Wind load .xlsx file and save as .csv
class Wind:
    def __init__(self):
        self.dataset = None

    def wind_raw_data_load(self, dataset_path, sht_num=0):
        self.dataset = pd.read_excel(dataset_path, sheet_name=sht_num)
        return self.dataset

# Original Weather load .xlsx file and save as .csv
class Weather:
    def __init__(self):
        self.dataset = None

    def weather_raw_data_load(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        return self.dataset

class WindWeather:
    def __init__(self):
        self.wind_dataset = None
        self.weather_dataset = None

    def wind_weather_dataset_load(self, wind_dataset_path, weather_dataset_path):
        self.wind_dataset = pd.read_csv(wind_dataset_path)
        self.weather_dataset = pd.read_csv(weather_dataset_path)
        return self.wind_dataset, self.weather_dataset


class ModelDataset:
    def __init__(self):
        self.dataset = None

    def dataset_load(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        return self.dataset