from features_wrap import Feature_Wraper

INIT_CASH = 1000000
INIT_STOCK = 0
DATA_PATH = "data/1min.csv"


class Gym:
    def __init__(self):
        self.total_step = 0
        self.fw = Feature_Wraper(DATA_PATH)

    def reset(self):
        self.cash = INIT_CASH
        self.stock = INIT_STOCK
        self.portfolio_value = INIT_CASH
        self.price = self.fw.prices[self.total_step]
        self.time_id = self.fw.time_id[self.total_step]
        state = [self.time_id, self.fw.features[self.total_step],
                 self.cash, self.stock]
        return state

    def step(self, action):
        if action == "short":
            self.cash += self.stock * self.price
            self.stock = 0
        elif action == "long":
            self.stock += self.cash//self.price
            self.cash = self.cash % self.price
        elif action == "hold":
            pass
        else:
            print("illegal action input")
            return None
        print(self.price)
        self.total_step += 1
        ts = self.total_step
        self.price = self.fw.prices[ts]
        pf_v = self.portfolio_value
        self.portfolio_value = self.cash + self.stock * self.price
        reward = self.portfolio_value - pf_v
        time_id = self.fw.time_id[ts]
        feature = self.fw.features[ts]
        state = [time_id, feature, self.cash, self.stock]
        return reward, state

if __name__ == "__main__":
    env = Gym()
    state = env.reset()
    print(env.step("long"))
    print(env.step("hold"))
    print(env.step("short"))