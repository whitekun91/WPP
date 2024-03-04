from datapreprocess.wind import *

if __name__ == '__main__':
    # Original wind dataset(.xlsx) path
    hk_dataset_path = './dataset/rawdata/wind/붙임1_한경풍력 발전실적 정보.xlsx'
    ss_dataset_path = './dataset/rawdata/wind/붙임2_성산풍력 발전실적 정보.xlsx'

    # Sheet Number
    HK1_AND_SS_SHT_NUM = 0
    HK2_SHT_NUM = 1

    # Convert .csv file and save
    WindRawDataset(Wind().wind_raw_data_load(hk_dataset_path, HK1_AND_SS_SHT_NUM)).dataset_to_csv('hk1')
    WindRawDataset(Wind().wind_raw_data_load(hk_dataset_path, HK2_SHT_NUM)).dataset_to_csv('hk2')
    WindRawDataset(Wind().wind_raw_data_load(ss_dataset_path, HK1_AND_SS_SHT_NUM)).dataset_to_csv('ss')

    # wind dataset(.csv) path
    hk1_csv_dataset_path = './dataset/csvdata/wind_raw_dataset/hk1_raw_data.csv'
    hk2_csv_dataset_path = './dataset/csvdata/wind_raw_dataset/hk2_raw_data.csv'
    ss_csv_dataset_path = './dataset/csvdata/wind_raw_dataset/ss_raw_data.csv'

    '''
     Load .csv file and preprocessing
    '''
    # Hankyung 1st power generation plant
    hk1_dataset = WindDataset(hk1_csv_dataset_path)
    hk1_dataset.time_split()
    hk1_dataset.generation_max_select()
    hk1_dataset.dummy_dataset_and_generation_dataset_merge('hk1')

    # Hankyung 2nd power generation plant
    hk2_dataset = WindDataset(hk2_csv_dataset_path)
    hk2_dataset.time_split()
    hk2_dataset.generation_max_select_transformed_kw()
    hk2_dataset.dummy_dataset_and_generation_dataset_merge('hk2')

    # Seongsan power generation plant
    ss_dataset = WindDataset(ss_csv_dataset_path)
    ss_dataset.time_split()
    ss_dataset.generation_max_select_transformed_kw()
    ss_dataset.dummy_dataset_and_generation_dataset_merge('ss')