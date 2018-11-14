'''
数据要求：
- min 级数据
- 数据齐全
- 第一列必须为 DateTime
'''
import os
import pickle
import itertools
import numpy as np
import pandas as pd

MIN_DAY = 240
WINDOW_SIZE = 10


class Feature_Wraper:
    def __init__(self, path):
        if not os.path.exists(path):
            print("data file does not exist")
            return None
        self.path = path
        self.time_id, self.prices, self.features = [], [], []
        self.load_data()

    def load_data(self):
        path = self.path + ".wb"
        if os.path.exists(path):
            with open(path, 'rb') as file:
                data = pickle.load(file)
                self.time_id, self.prices, self.features = data[0], data[1], data[2]
        else:
            self.read_csv_2()
            with open(path, 'wb') as file:
                pickle.dump([self.time_id, self.prices, self.features], file)

    def read_csv(self):
        df = pd.read_csv(self.path)
        for day in range(len(df)//MIN_DAY):
            for i in range(MIN_DAY-WINDOW_SIZE+1):
                today_time = i + WINDOW_SIZE
                total_time = day * (MIN_DAY - WINDOW_SIZE) + today_time
                index = day * MIN_DAY + i
                _df = df[index: index+10]
                self.time_id.append(total_time)
                self.prices.append(self.data2prices(_df))
                self.features.append(self.data2feature(_df, today_time))

    def data2prices(self, df):
        price = df.tail(1)["close"].values
        return price

    def data2feature(self, df, time):
        np_ndarray = df.values
        '''
            add some feature progress here
        '''
        np_ndarray[:, 0] = 1
        np_ndarray = np_ndarray / (np_ndarray.max(axis=0)+0.0001)
        for i in range(WINDOW_SIZE):
            np_ndarray[i][0] = time - WINDOW_SIZE + 1 + i
        return np_ndarray.astype(float)

    def read_csv_2(self):
        df = pd.read_csv(self.path)

        df['MDMin'] = df['MDTime'].apply(lambda x: int(x // 1e5))
        minute_data = df.groupby('MDMin').first().reset_index().drop('MDTime', axis=1)

        features_data = self.cal_feature(minute_data)

        self.time_id = features_data.iloc[:1 - WINDOW_SIZE, 0].tolist()
        for i in features_data.index[:1 - WINDOW_SIZE]:
            _df = features_data[i:i+WINDOW_SIZE]
            self.prices.append(_df.tail(1)['LastPx'].values)
            features = _df.values
            features = features / (features.max(axis=0) + 0.0001)
            self.features.append(features)

    def cal_feature(self, minute_data):
        fields = [ 'MDMin', 'NumTrades', 'PreVolume', 'TotalValueTrade', 'LastPx', 
                   'TotalBidQty', 'TotalOfferQty', 'WeightedAvgBidPx', 'WeightedAvgOfferPx']
        pankou_fileds = list(map(lambda x: ''.join(x), itertools.product(('Sell', 'Buy'), ('%d'%(i+1) for i in range(5)), ('Price', 'OrderQty'))))
        fields.extend(pankou_fileds)

        features_data = minute_data.loc[:, fields]

        features_data.drop(features_data[features_data['WeightedAvgBidPx'] == 0].index, inplace=True)
        features_data.drop(features_data[features_data['WeightedAvgOfferPx'] == 0].index, inplace=True)
        features_data['WeightedAvgBidPx'] = features_data['LastPx'] - features_data['WeightedAvgBidPx']
        features_data['WeightedAvgOfferPx'] = features_data['WeightedAvgOfferPx'] - features_data['LastPx']
        for field in pankou_fileds:
            if 'Price' in field:
                features_data[field] = (features_data[field] - features_data['LastPx']) * (1 if field[0] == 'S' else -1)

        features_data['PreValueTrade'] = features_data['TotalValueTrade'].diff()
        features_data['PreNumTrades'] = features_data['NumTrades'].diff()
        features_data.drop(['TotalValueTrade', 'NumTrades'], axis=1, inplace=True)

        features_data['ma_5'] = features_data['LastPx'].rolling(5).mean()
        features_data['std_5'] = features_data['LastPx'].rolling(5).std()
        features_data['ma_10'] = features_data['LastPx'].rolling(10).mean()
        features_data['std_10'] = features_data['LastPx'].rolling(10).std()
        features_data['rt'] =  features_data['LastPx'].pct_change()

        features_data['ma_volume_5'] = features_data['PreVolume'].rolling(5).mean()
        features_data['ma_volume_10'] = features_data['PreVolume'].rolling(10).mean()

        features_data['highest_10'] = features_data['LastPx'].rolling(10).max()
        features_data['lowest_10'] = features_data['LastPx'].rolling(10).min()

        features_data['MDMin'] = range(1, 1 + len(features_data))

        return features_data.dropna()


if __name__ == "__main__":
#    fw = Feature_Wraper("data/XSHE_2015_2018.csv")
    fw = Feature_Wraper("data/20180621_000001_scale.csv")
    print(fw.prices[0])
    print(fw.prices[-1])
    print(len(fw.time_id) == len(fw.features) == len(fw.prices))
