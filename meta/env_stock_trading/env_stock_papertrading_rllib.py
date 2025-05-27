import datetime
import threading
import time

import alpaca_trade_api as tradeapi
import gym
import numpy as np
import pandas as pd

from meta.data_processors.alpaca import Alpaca


class StockEnvEmpty(gym.Env):
    def __init__(self, config):
        state_dim = config["state_dim"]
        action_dim = config["action_dim"]
        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )

    def reset(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, actions):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, {}


class AlpacaPaperTrading_rllib:
    def __init__(
        self,
        ticker_list,
        time_interval,
        agent,
        cwd,
        net_dim,
        state_dim,
        action_dim,
        API_KEY,
        API_SECRET,
        API_BASE_URL,
        tech_indicator_list,
        turbulence_thresh=30,
        max_stock=1e2,
    ):
        print("agent", agent)
        if agent == "ppo":
            from ray.rllib.agents import ppo
            from ray.rllib.agents.ppo.ppo import PPOTrainer

            config = ppo.DEFAULT_CONFIG.copy()
            config["env"] = StockEnvEmpty
            config["log_level"] = "WARN"
            config["env_config"] = {"state_dim": state_dim, "action_dim": action_dim}
            trainer = PPOTrainer(env=StockEnvEmpty, config=config)
            try:
                trainer.restore(cwd)
                self.agent = trainer
                print("Restoring from checkpoint path", cwd)
            except Exception as e:
                raise RuntimeError(f"Failed to load RLlib agent: {e}")
        else:
            raise ValueError("Agent input is NOT supported yet.")

        try:
            self.alpaca = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, "v2")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Alpaca API: {e}")

        intervals = {"1s": 1, "5s": 5, "1Min": 60, "5Min": 300, "15Min": 900}
        if time_interval not in intervals:
            raise ValueError("Unsupported time interval.")
        self.time_interval = intervals[time_interval]

        self.tech_indicator_list = tech_indicator_list
        self.turbulence_thresh = turbulence_thresh
        self.max_stock = max_stock
        self.stocks = np.asarray([0] * len(ticker_list))
        self.stocks_cd = np.zeros_like(self.stocks)
        self.cash = None
        self.stocks_df = pd.DataFrame(self.stocks, columns=["stocks"], index=ticker_list)
        self.asset_list = []
        self.price = np.asarray([0] * len(ticker_list))
        self.stockUniverse = ticker_list
        self.turbulence_bool = 0
        self.equities = []

    def run(self):
        try:
            orders = self.alpaca.list_orders(status="open")
            for order in orders:
                self.alpaca.cancel_order(order.id)
        except Exception as e:
            print(f"[Cancel Orders Error] {e}")

        print("Waiting for market to open...")
        try:
            tAMO = threading.Thread(target=self.awaitMarketOpen)
            tAMO.start()
            tAMO.join(timeout=1800)
        except Exception as e:
            print(f"[Await Market Open Error] {e}")
        print("Market opened.")

        while True:
            try:
                clock = self.alpaca.get_clock()
                closingTime = clock.next_close.replace(tzinfo=datetime.timezone.utc).timestamp()
                currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
                self.timeToClose = closingTime - currTime
            except Exception as e:
                print(f"[Clock Error] {e}")
                self.timeToClose = 60

            if self.timeToClose < 60:
                print("Market closing soon. Stop trading.")
                break

            try:
                trade = threading.Thread(target=self.trade)
                trade.start()
                trade.join(timeout=60)

                account = self.alpaca.get_account()
                last_equity = float(account.last_equity) if hasattr(account, "last_equity") else 0
                self.equities.append([time.time(), last_equity])
                np.save("paper_trading_records.npy", np.asarray(self.equities, dtype=float))
            except Exception as e:
                print(f"[Trade Cycle Error] {e}")

            time.sleep(self.time_interval)

    def awaitMarketOpen(self):
        while True:
            try:
                clock = self.alpaca.get_clock()
                if clock.is_open:
                    return
                openingTime = clock.next_open.replace(tzinfo=datetime.timezone.utc).timestamp()
                currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
                timeToOpen = int((openingTime - currTime) / 60)
                print(f"{timeToOpen} minutes til market open.")
                time.sleep(60)
            except Exception as e:
                print(f"[Market Open Wait Error] {e}")
                time.sleep(60)

    def trade(self):
        try:
            state = self.get_state()
            action = self.agent.compute_single_action(state)
            action = np.nan_to_num(action * self.max_stock).astype(int)
        except Exception as e:
            print(f"[Action Error] {e}")
            return

        self.stocks_cd += 1
        if self.turbulence_bool == 0:
            min_action = 10
            for index in np.where(action < -min_action)[0]:
                try:
                    sell_num_shares = min(self.stocks[index], -action[index])
                    qty = abs(int(sell_num_shares))
                    respSO = []
                    t = threading.Thread(
                        target=self.submitOrder,
                        args=(qty, self.stockUniverse[index], "sell", respSO),
                    )
                    t.start()
                    t.join(timeout=30)
                    self.cash = float(self.alpaca.get_account().cash)
                    self.stocks_cd[index] = 0
                except Exception as e:
                    print(f"[Sell Error] {e}")

            for index in np.where(action > min_action)[0]:
                try:
                    tmp_cash = max(self.cash, 0)
                    buy_num_shares = min(tmp_cash // self.price[index], abs(int(action[index])))
                    qty = abs(int(buy_num_shares))
                    respSO = []
                    t = threading.Thread(
                        target=self.submitOrder,
                        args=(qty, self.stockUniverse[index], "buy", respSO),
                    )
                    t.start()
                    t.join(timeout=30)
                    self.cash = float(self.alpaca.get_account().cash)
                    self.stocks_cd[index] = 0
                except Exception as e:
                    print(f"[Buy Error] {e}")
        else:
            try:
                positions = self.alpaca.list_positions()
                for position in positions:
                    side = "sell" if position.side == "long" else "buy"
                    qty = abs(int(float(position.qty)))
                    respSO = []
                    t = threading.Thread(
                        target=self.submitOrder, args=(qty, position.symbol, side, respSO)
                    )
                    t.start()
                    t.join(timeout=30)
                self.stocks_cd[:] = 0
            except Exception as e:
                print(f"[Turbulence Liquidation Error] {e}")

    def get_state(self):
        try:
            alpaca = Alpaca(api=self.alpaca)
            price, tech, turbulence = alpaca.fetch_latest_data(
                ticker_list=self.stockUniverse,
                time_interval="1Min",
                tech_indicator_list=self.tech_indicator_list,
            )
            self.turbulence_bool = 1 if turbulence >= self.turbulence_thresh else 0
            turbulence = (
                self.sigmoid_sign(turbulence, self.turbulence_thresh) * 2**-5
            ).astype(np.float32)
            tech = tech * 2**-7

            stocks = [0] * len(self.stockUniverse)
            for position in self.alpaca.list_positions():
                try:
                    ind = self.stockUniverse.index(position.symbol)
                    stocks[ind] = abs(int(float(position.qty)))
                except ValueError:
                    continue

            self.stocks = np.asarray(stocks, dtype=float)
            self.price = price
            self.cash = float(self.alpaca.get_account().cash)

            amount = np.array(max(self.cash, 1e4) * (2**-12), dtype=np.float32)
            scale = np.array(2**-6, dtype=np.float32)
            state = np.hstack((
                amount,
                turbulence,
                self.turbulence_bool,
                price * scale,
                self.stocks * scale,
                self.stocks_cd,
                tech,
            )).astype(np.float32)
            print(len(self.stockUniverse))
            return state
        except Exception as e:
            print(f"[Get State Error] {e}")
            return np.zeros(3 * len(self.stockUniverse) + len(self.tech_indicator_list) + 3, dtype=np.float32)

    def submitOrder(self, qty, stock, side, resp):
        if qty > 0:
            try:
                self.alpaca.submit_order(stock, qty, side, "market", "day")
                print(f"Market order of | {qty} {stock} {side} | completed.")
                resp.append(True)
            except Exception as e:
                print(f"Order of | {qty} {stock} {side} | failed: {e}")
                resp.append(False)
        else:
            print(f"Quantity is 0, order of | {qty} {stock} {side} | skipped.")
            resp.append(True)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5
        try:
            return sigmoid(ary / thresh) * thresh
        except Exception as e:
            print(f"[Sigmoid Error] {e}")
            return np.zeros_like(ary)
