from datapreprocess.weather import *

if __name__ == '__main__':

    '''
         Load .csv file and preprocessing
    '''

    # weather dataset(.csv) path list
    locations = ['hankyeong', 'seongsan']
    years = ['2014', '2015', '2016', '2017']
    weather_types = ['temp3h', 'rain6h', 'snow6h', 'humidity', 'rainprobability', 'raintype', 'seawave', 'skystatus',
                    'windspeed', 'winddirection']

    for loca in locations:
        for year in years:
            temp_dataset = []
            for w_type in weather_types:
                data_path = './dataset/rawdata/weather/{}/{}/weather_forecast_{}.csv'.format(loca, year, w_type)
                globals()[w_type] = WeatherDataset(Weather().weather_raw_data_load(data_path))
                globals()[w_type].data_filtering(w_type)
                globals()['{}_dataset'.format(w_type)] = globals()[w_type].data_datetime_setting(year)
                temp_dataset.append(globals()['{}_dataset'.format(w_type)])
            WeatherMerging().data_merging(temp_dataset, loca, year)


    for loca in locations:
        weather_2014_path = './dataset/csvdata/weather_dataset/{}/2014/weather_forecast_2014_4hours_target.csv'.format(
            loca)
        weather_2015_path = './dataset/csvdata/weather_dataset/{}/2015/weather_forecast_2015_4hours_target.csv'.format(
            loca)
        weather_2016_path = './dataset/csvdata/weather_dataset/{}/2016/weather_forecast_2016_4hours_target.csv'.format(
            loca)
        weather_2017_path = './dataset/csvdata/weather_dataset/{}/2017/weather_forecast_2017_4hours_target.csv'.format(
            loca)

        WeatherMerging().dataset_merging_deleting_null_value(weather_2014_path, weather_2015_path, weather_2016_path,
                                                             weather_2017_path, loca)


