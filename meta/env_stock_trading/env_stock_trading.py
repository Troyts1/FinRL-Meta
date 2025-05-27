import gym
import numpy as np
from numpy import random as rd

class StockTradingEnv(gym.Env):
    def __init__(
        self,
        config,
        initial_account=1e6,
        gamma=0.99,
        turbulence_thresh=99,
        min_stock_rate=0.1,
        max_stock=1e2,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=2**-11,
        initial_stocks=None,
    ):
        try:
            self.price_array = config["price_array"].astype(np.float32)
            self.tech_array = config["tech_array"].astype(np.float32) * 2**-7
            self.turbulence_array_raw = config["turbulence_array"]
            self.if_train = config["if_train"]
        except KeyError as e:
            raise ValueError(f"Missing config parameter: {e}")

        self.turbulence_bool = (self.turbulence_array_raw > turbulence_thresh).astype(np.float32)
        self.turbulence_array = (
            self.sigmoid_sign(self.turbulence_array_raw, turbulence_thresh) * 2**-5
        ).astype(np.float32)

        stock_dim = self.price_array.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = np.zeros(stock_dim, dtype=np.float32) if initial_stocks is None else initial_stocks

        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_array.shape[1]
        self.action_dim = stock_dim
        self.max_step = self.price_array.shape[0] - 1

        self.env_name = "StockEnv"
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0

        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        try:
            self.time = 0
            price = self.price_array[self.time]

            if self.if_train:
                self.stocks = (self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)).astype(np.float32)
                self.stocks_cool_down = np.zeros_like(self.stocks)
                self.cash = self.initial_capital * rd.uniform(0.95, 1.05) - (self.stocks * price).sum()
            else:
                self.stocks = self.initial_stocks.astype(np.float32)
                self.stocks_cool_down = np.zeros_like(self.stocks)
                self.cash = self.initial_capital

            self.total_asset = self.cash + (self.stocks * price).sum()
            self.initial_total_asset = self.total_asset
            self.gamma_reward = 0.0

            return np.array(self.get_state(price), dtype=np.float32)
        except Exception as e:
            print(f"[Reset Error] {e}")
            return np.zeros(self.state_dim, dtype=np.float32)

    def step(self, actions):
        try:
            actions = np.nan_to_num(actions * self.max_stock).astype(int)
            self.time += 1
            if self.time >= len(self.price_array):
                print(f"[Step Warning] Time {self.time} exceeds array length.")
                self.time = self.max_step

            price = self.price_array[self.time]
            self.stocks_cool_down += 1

            if self.turbulence_bool[self.time] == 0:
                min_action = int(self.max_stock * self.min_stock_rate)
                for index in np.where(actions < -min_action)[0]:
                    if 0 <= index < len(price) and price[index] > 0:
                        sell_num_shares = min(self.stocks[index], -actions[index])
                        self.stocks[index] -= sell_num_shares
                        self.cash += price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                        self.stocks_cool_down[index] = 0

                for index in np.where(actions > min_action)[0]:
                    if 0 <= index < len(price) and price[index] > 0:
                        buy_num_shares = min(self.cash // price[index], actions[index])
                        self.stocks[index] += buy_num_shares
                        self.cash -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                        self.stocks_cool_down[index] = 0
            else:
                self.cash += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
                self.stocks[:] = 0
                self.stocks_cool_down[:] = 0

            state = self.get_state(price)
            total_asset = self.cash + (self.stocks * price).sum()
            reward = np.float32((total_asset - self.total_asset) * self.reward_scaling)
            self.total_asset = total_asset

            self.gamma_reward = self.gamma_reward * self.gamma + reward
            done = self.time == self.max_step
            if done:
                reward = self.gamma_reward
                self.episode_return = total_asset / self.initial_total_asset

            return np.array(state, dtype=np.float32), reward, done, {}
        except Exception as e:
            print(f"[Step Error] {e}")
            return np.zeros(self.state_dim, dtype=np.float32), -1.0, True, {"error": str(e)}

    def get_state(self, price):
        try:
            cash = np.array(self.cash * (2**-12), dtype=np.float32)
            scale = np.array(2**-6, dtype=np.float32)
            return np.hstack((
                cash,
                self.turbulence_array[self.time],
                self.turbulence_bool[self.time],
                price * scale,
                self.stocks * scale,
                self.stocks_cool_down,
                self.tech_array[self.time],
            )).astype(np.float32)
        except Exception as e:
            print(f"[State Error] {e}")
            return np.zeros(self.state_dim, dtype=np.float32)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5
        try:
            return sigmoid(ary / thresh) * thresh
        except Exception as e:
            print(f"[Sigmoid Error] {e}")
            return np.zeros_like(ary)
