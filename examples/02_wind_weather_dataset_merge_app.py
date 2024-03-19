from datamerging.windweathermerging import *

if __name__ =='__main__':
    locations = ['hankyeong', 'seongsan']
    hk_power_plants = ['hk1', 'hk2']

    for loc in locations:
        weather_dataset_path = './dataset/csvdata/weather_dataset/weather_forecast_{}_4hours.csv'.format(loc)
        if loc == 'hankyeong':
            for plant in hk_power_plants:
                if plant == 'hk1':
                    wind_dataset_path = './dataset/csvdata/wind_dataset/wind_generation_{}_dataset.csv'.format(plant)
                    wind_dataset, wether_dataset = WindWeather().wind_weather_dataset_load(wind_dataset_path, weather_dataset_path)
                    Merging(wind_dataset, wether_dataset).data_merge(plant)
                else:
                    wind_dataset_path = './dataset/csvdata/wind_dataset/wind_generation_{}_dataset.csv'.format(plant)
                    wind_dataset, wether_dataset = WindWeather().wind_weather_dataset_load(wind_dataset_path,
                                                                                           weather_dataset_path)
                    Merging(wind_dataset, wether_dataset).data_merge(plant)
        else:
            wind_dataset_path = './dataset/csvdata/wind_dataset/wind_generation_ss_dataset.csv'
            wind_dataset, wether_dataset = WindWeather().wind_weather_dataset_load(wind_dataset_path,
                                                                                   weather_dataset_path)
            Merging(wind_dataset, wether_dataset).data_merge('ss')