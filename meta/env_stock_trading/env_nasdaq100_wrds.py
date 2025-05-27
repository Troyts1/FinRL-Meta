import os
import gym
import numpy as np
from numpy import random as rd

gym.logger.set_level(40)

class StockEnvNAS100:
    def __init__(
        self,
        cwd="./data/nas100",
        price_ary=None,
        tech_ary=None,
        turbulence_ary=None,
        gamma=0.999,
        turbulence_thresh=30,
        min_stock_rate=0.1,
        max_stock=1e2,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        data_gap=4,
        reward_scaling=2**-11,
        ticker_list=None,
        tech_indicator_list=None,
        initial_stocks=None,
        if_eval=False,
        if_trade=False,
    ):
        self.min_stock_rate = min_stock_rate
        beg_i, mid_i, end_i = 0, int(211210), int(422420)
        (i0, i1) = (beg_i, mid_i) if if_eval else (mid_i, end_i)

        try:
            data_arrays = (
                self.load_data(cwd) if cwd is not None else (price_ary, tech_ary, turbulence_ary)
            )
            if not if_trade:
                data_arrays = [ary[i0:i1:data_gap] for ary in data_arrays]
            else:
                data_arrays = [ary[int(422420):int(528026):data_gap] for ary in data_arrays]

            self.price_ary, self.tech_ary, turbulence_ary = data_arrays
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load or slice arrays: {e}")

        self.tech_ary = self.tech_ary * 2**-7
        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (
            self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2**-5
        ).astype(np.float32)

        try:
            stock_dim = self.price_ary.shape[1]
        except Exception as e:
            raise RuntimeError(f"[ERROR] price_ary missing shape[1]: {e}")

        self.gamma = gamma
        self.max_stock = max_stock
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        self.env_name = "StockEnvNAS"
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_discrete = False
        self.target_return = 2.2
        self.episode_return = 0.0

    def reset(self):
        try:
            self.day = 0
            price = self.price_ary[self.day]

            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cd = np.zeros_like(self.stocks)
            self.amount = (
                self.initial_capital * rd.uniform(0.95, 1.05) - (self.stocks * price).sum()
            )

            self.total_asset = self.amount + (self.stocks * price).sum()
            self.initial_total_asset = self.total_asset
            self.gamma_reward = 0.0
            return self.get_state(price)
        except Exception as e:
            print(f"[RESET ERROR] {e}")
            return np.zeros(self.state_dim, dtype=np.float32)

    def step(self, actions):
        try:
            actions = np.nan_to_num(actions * self.max_stock).astype(int)
            self.day += 1

            if self.day >= len(self.price_ary):
                print(f"[STEP WARNING] Day {self.day} exceeds price_ary length {len(self.price_ary)}")
                self.day = self.max_step

            price = self.price_ary[self.day]
            self.stocks_cd += 1

            if self.turbulence_bool[self.day] == 0:
                min_action = int(self.max_stock * self.min_stock_rate)
                for index in np.where(actions < -min_action)[0]:
                    if 0 <= index < len(price) and price[index] > 0:
                        sell_num_shares = min(self.stocks[index], -actions[index])
                        self.stocks[index] -= sell_num_shares
                        self.amount += price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                        self.stocks_cd[index] = 0
                for index in np.where(actions > min_action)[0]:
                    if 0 <= index < len(price) and price[index] > 0:
                        buy_num_shares = min(self.amount // price[index], actions[index])
                        self.stocks[index] += buy_num_shares
                        self.amount -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                        self.stocks_cd[index] = 0
            else:
                self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
                self.stocks[:] = 0
                self.stocks_cd[:] = 0

            state = self.get_state(price)
            total_asset = self.amount + (self.stocks * price).sum()
            reward = (total_asset - self.total_asset) * self.reward_scaling
            self.total_asset = total_asset

            self.gamma_reward = self.gamma_reward * self.gamma + reward
            done = self.day == self.max_step
            if done:
                reward = self.gamma_reward
                self.episode_return = total_asset / self.initial_total_asset

            return state, reward, done, dict()
        except Exception as e:
            print(f"[STEP ERROR] {e}")
            fallback_state = np.nan_to_num(self.get_state(np.ones_like(self.price_ary[0])))
            return fallback_state, -1.0, True, dict(error=str(e))

    def get_state(self, price):
        try:
            amount = np.array(max(self.amount, 1e4) * (2**-12), dtype=np.float32)
            scale = np.array(2**-6, dtype=np.float32)
            return np.hstack((
                amount,
                self.turbulence_ary[self.day],
                self.turbulence_bool[self.day],
                price * scale,
                self.stocks * scale,
                self.stocks_cd,
                self.tech_ary[self.day],
            ))
        except Exception as e:
            print(f"[STATE ERROR] {e}")
            return np.zeros(self.state_dim, dtype=np.float32)

    def load_data(self, cwd):
        try:
            p_path = f"{cwd}/price_ary.npy"
            t_path = f"{cwd}/tech_ary.npy"
            turb_path = f"{cwd}/turb_ary.npy"

            assert os.path.exists(p_path), f"Missing {p_path}"
            assert os.path.exists(t_path), f"Missing {t_path}"
            assert os.path.exists(turb_path), f"Missing {turb_path}"

            price_ary = np.load(p_path).astype(np.float32)
            tech_ary = np.load(t_path).astype(np.float32)
            turbulence_ary = np.load(turb_path)
            turbulence_ary = turbulence_ary.repeat(390)[-528026:]
            return price_ary, tech_ary, turbulence_ary
        except Exception as e:
            raise RuntimeError(f"[DATA LOAD ERROR] {e}")

    def draw_cumulative_return(self, args, _torch):
        try:
            state_dim = self.state_dim
            action_dim = self.action_dim

            agent = args.agent
            net_dim = args.net_dim
            cwd = args.cwd

            agent.init(net_dim, state_dim, action_dim)
            agent.save_load_model(cwd=cwd, if_save=False)
            act = agent.act
            device = agent.device

            state = self.reset()
            episode_returns = []
            with _torch.no_grad():
                for i in range(self.max_step):
                    s_tensor = _torch.as_tensor((state,), device=device)
                    a_tensor = act(s_tensor)
                    action = a_tensor.detach().cpu().numpy()[0]
                    state, reward, done, _ = self.step(action)
                    total_asset = self.amount + (self.price_ary[self.day] * self.stocks).sum()
                    episode_return = total_asset / self.initial_total_asset
                    episode_returns.append(episode_return)
                    if done:
                        break

            import matplotlib.pyplot as plt
            plt.plot(episode_returns)
            plt.grid()
            plt.title("cumulative return")
            plt.xlabel("day")
            plt.ylabel("multiple of initial account")
            plt.savefig(f"{cwd}/cumulative_return.jpg")
            print(f"| draw_cumulative_return: saved in {cwd}/cumulative_return.jpg")
            return episode_returns
        except Exception as e:
            print(f"[DRAW RETURN ERROR] {e}")
            return []

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5
        try:
            return sigmoid(ary / thresh) * thresh
        except Exception as e:
            print(f"[SIGMOID ERROR] {e}")
            return np.zeros_like(ary)
