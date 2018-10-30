'''
数据要求：
- min 级数据
- 数据齐全
- 第一列必须为 DateTime
'''
import os
import pickle
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
            self.read_csv()
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


if __name__ == "__main__":
    fw = Feature_Wraper("data/XSHE_2015_2018.csv")
    print(fw.prices[0])
    print(fw.prices[-1])
    print(len(fw.time_id) == len(fw.features) == len(fw.prices))
