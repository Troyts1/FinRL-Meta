import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv


class StockTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        make_plots=False,
        print_verbosity=2,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        initial_buy=False,
        hundred_each_trade=True,
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,))
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.initial_buy = initial_buy
        self.hundred_each_trade = hundred_each_trade

        self.data = self.df.loc[self.day, :] if day in self.df.index else self.df.iloc[0]
        self.terminal = False

        self.state = self._initiate_state()
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0

        self.portfolio_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self._seed()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            if self.make_plots:
                try:
                    self._make_plot()
                except Exception as e:
                    print(f"[Plot Error] {e}")

            portfolio_df = self.get_portfolio_df()
            begin_total_asset = portfolio_df["prev_total_asset"].iloc[0]
            end_total_asset = portfolio_df["total_asset"].iloc[-1]
            tot_reward = end_total_asset - begin_total_asset

            portfolio_df["daily_return"] = portfolio_df["total_asset"].pct_change(1)
            sharpe = None
            if portfolio_df["daily_return"].std() != 0:
                sharpe = (252**0.5) * portfolio_df["daily_return"].mean() / portfolio_df["daily_return"].std()

            if self.episode % self.print_verbosity == 0:
                print(f"EP {self.episode} — Start: {begin_total_asset:.2f}, End: {end_total_asset:.2f}, Reward: {tot_reward:.2f}, Cost: {self.cost:.2f}, Trades: {self.trades}, Sharpe: {sharpe:.3f}" if sharpe else "")

            if self.model_name and self.mode:
                try:
                    df_actions = self.save_action_memory()
                    df_actions.to_csv(f"results/actions_{self.mode}_{self.model_name}_{self.episode}.csv")
                    portfolio_df.to_csv(f"results/portfolio_{self.mode}_{self.model_name}_{self.episode}.csv", index=False)
                except Exception as e:
                    print(f"[CSV Save Error] {e}")

            return self.state, self.reward, self.terminal, {}

        actions = np.nan_to_num(actions * self.hmax).astype(int)
        if self.turbulence_threshold and self.turbulence >= self.turbulence_threshold:
            actions = np.array([-self.hmax] * self.stock_dim)

        begin_cash = self.state[0]
        begin_market_value = self._get_market_value()
        begin_total_asset = begin_cash + begin_market_value
        begin_cost = self.cost
        begin_trades = self.trades
        begin_stock = self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]

        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

        for index in sell_index:
            actions[index] = self._sell_stock(index, actions[index]) * (-1)
        for index in buy_index:
            actions[index] = self._buy_stock(index, actions[index])

        if self.turbulence_threshold is not None and "turbulence" in self.data.columns:
            self.turbulence = self.data["turbulence"].values[0]

        end_cash = self.state[0]
        end_market_value = self._get_market_value()
        end_total_asset = end_cash + end_market_value
        end_cost = self.cost
        end_trades = self.trades
        end_stock = self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]

        self.actions_memory.append(actions)
        self.reward = end_total_asset - begin_total_asset
        for i in range(self.stock_dim):
            if begin_stock[i] == end_stock[i]:
                self.reward -= (self.state[i + 1] * self.state[self.stock_dim + 1 + i]) * 0.001

        self.reward *= self.reward_scaling
        self.portfolio_memory.append({
            "date": self._get_date(),
            "prev_total_asset": begin_total_asset,
            "prev_cash": begin_cash,
            "prev_market_value": begin_market_value,
            "total_asset": end_total_asset,
            "cash": end_cash,
            "market_value": end_market_value,
            "cost": end_cost - begin_cost,
            "trades": end_trades - begin_trades,
            "reward": self.reward,
        })
        self.date_memory.append(self._get_date())

        self.day += 1
        self.data = self.df.loc[self.day, :] if self.day in self.df.index else self.df.iloc[-1]
        self.state = self._update_state()
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.day = 0
        self.data = self.df.loc[self.day, :] if self.day in self.df.index else self.df.iloc[0]
        self.state = self._initiate_state()
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.portfolio_memory = []
        self.episode += 1
        return self.state

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        try:
            if self.initial:
                if len(self.df.tic.unique()) > 1:
                    state = [self.initial_amount] + self.data.close.values.tolist() + [0] * self.stock_dim + sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
                    return self.initial_buy_() if self.initial_buy else state
                else:
                    return [self.initial_amount] + [self.data.close] + [0] * self.stock_dim + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
            else:
                if len(self.df.tic.unique()) > 1:
                    return [self.previous_state[0]] + self.data.close.values.tolist() + self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)] + sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
                else:
                    return [self.previous_state[0]] + [self.data.close] + self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)] + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
        except Exception as e:
            print(f"[Initiate State Error] {e}")
            return np.zeros(self.state_space, dtype=np.float32)

    def _update_state(self):
        try:
            if len(self.df.tic.unique()) > 1:
                return [self.state[0]] + self.data.close.values.tolist() + list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]) + sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
            else:
                return [self.state[0]] + [self.data.close] + list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]) + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
        except Exception as e:
            print(f"[Update State Error] {e}")
            return np.zeros(self.state_space, dtype=np.float32)

    def _get_date(self):
        try:
            return self.data.date.unique()[0] if len(self.df.tic.unique()) > 1 else self.data.date
        except Exception:
            return "1970-01-01"

    def _sell_stock(self, index, action):
        try:
            if self.turbulence_threshold and self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0 and self.state[index + self.stock_dim + 1] > 0:
                    sell_num = self.state[index + self.stock_dim + 1]
                    amount = self.state[index + 1] * sell_num
                    cost = amount * self.sell_cost_pct
                    self.state[0] += amount - cost
                    self.state[index + self.stock_dim + 1] = 0
                    self.cost += cost
                    self.trades += 1
                    return sell_num
                return 0
            if self.state[index + 1] > 0 and self.state[index + self.stock_dim + 1] > 0:
                sell_num = min(abs(action), self.state[index + self.stock_dim + 1])
                if self.hundred_each_trade:
                    sell_num = (sell_num // 100) * 100
                amount = self.state[index + 1] * sell_num
                cost = amount * self.sell_cost_pct
                self.state[0] += amount - cost
                self.state[index + self.stock_dim + 1] -= sell_num
                self.cost += cost
                self.trades += 1
                return sell_num
            return 0
        except Exception as e:
            print(f"[Sell Error] {e}")
            return 0

    def _buy_stock(self, index, action):
        try:
            if self.turbulence_threshold and self.turbulence >= self.turbulence_threshold:
                return 0
            if self.state[index + 1] > 0:
                available_amount = self.state[0] // self.state[index + 1]
                buy_num = min(available_amount, action)
                if self.hundred_each_trade:
                    buy_num = (buy_num // 100) * 100
                if buy_num > 0:
                    amount = self.state[index + 1] * buy_num
                    cost = amount * self.buy_cost_pct
                    self.state[0] -= amount + cost
                    self.state[index + self.stock_dim + 1] += buy_num
                    self.cost += cost
                    self.trades += 1
                    return buy_num
            return 0
        except Exception as e:
            print(f"[Buy Error] {e}")
            return 0

    def _make_plot(self):
        try:
            df = self.get_portfolio_df()
            plt.plot(df["date"], df["total_asset"], color="r")
            plt.savefig(f"results/account_value_trade_{self.episode}.png")
            plt.close()
        except Exception as e:
            print(f"[Plot Error] {e}")

    def get_portfolio_df(self):
        df = pd.DataFrame(self.portfolio_memory)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")

    def _get_total_asset(self):
        return self.state[0] + self._get_market_value()

    def _get_market_value(self):
        return float(np.dot(self.state[1:(self.stock_dim + 1)], self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]))

    def save_asset_memory(self):
        return self.get_portfolio_df()[["date", "total_asset"]].rename(columns={"total_asset": "account_value"})

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            df_actions = pd.DataFrame(self.actions_memory, index=self.date_memory[:-1])
            df_actions.columns = self.data.tic.values
            df_actions.index.name = "date"
            return df_actions
        return pd.DataFrame({"date": self.date_memory[:-1], "actions": self.actions_memory})

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        return DummyVecEnv([lambda: self]), self.reset()

    def initial_buy_(self):
        try:
            prices = self.data.close.values.tolist()
            mv_per_tic = 0.5 * self.initial_amount // len(prices)
            buy_nums = np.array([mv_per_tic // p for p in prices], dtype=int)
            if self.hundred_each_trade:
                buy_nums = (buy_nums // 100) * 100
            total_buy_amount = np.sum(np.array(prices) * buy_nums)
            return [self.initial_amount - total_buy_amount] + prices + list(buy_nums) + sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
        except Exception as e:
            print(f"[Initial Buy Error] {e}")
            return self._initiate_state()
