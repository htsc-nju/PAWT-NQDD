import numpy as np
from features_wrap import Feature_Wraper

INIT_CASH = np.array([1000000.0])
INIT_STOCK = np.array([0.0])
DATA_PATH = "data/XSHE_2015_2018.csv"


class Gym:
    def __init__(self):
        self.total_step = 0
        self.fw = Feature_Wraper(DATA_PATH)
        self.DATA_SET_MAX = len(self.fw.time_id) - 3

    def reset(self):
        self.cash = INIT_CASH
        self.stock = INIT_STOCK
        self.portfolio_value = INIT_CASH
        self.price = self.fw.prices[self.total_step]
        self.time_id = self.fw.time_id[self.total_step]
        state = self.state_wrap(
            self.fw.features[self.total_step], self.cash, self.stock)
        return state

    def step(self, action):
        if action[0] == 1:
            self.cash += self.stock * self.price
            self.stock = 0
        elif action[1] == 1:
            self.stock += self.cash//self.price
            self.cash = self.cash % self.price
        elif action[2] == 1:
            pass
        else:
            print("illegal action input")
            return None
        self.total_step += 1
        ts = self.total_step
        self.price = self.fw.prices[ts]
        pf_v = self.portfolio_value
        self.portfolio_value = self.cash + self.stock * self.price
        reward = self.portfolio_value - pf_v
        time_id = self.fw.time_id[ts]
        feature = self.fw.features[ts]
        if self.total_step % 2000 == 0:
            print(time_id, self.cash, self.stock, self.price)
        state = self.state_wrap(feature, self.cash, self.stock)
        done = False
        if self.portfolio_value < self.price:
            done = True
        info_nodata = False
        if ts > self.DATA_SET_MAX:
            info_nodata = True
        return state, reward, done, info_nodata

    def state_wrap(self, features, cash, stock):
        l = len(features)
        return np.concatenate((features, np.ones((l, 1))*cash, np.ones((l, 1))*stock), axis=1).astype(float)


if __name__ == "__main__":
    env = Gym()
    state = env.reset()
    env.step([0, 1, 0])
    # print(env.step([0, 1, 0]))
    # print(env.step([0, 0, 1]))
    # print(env.step([1, 0, 0]))
