import datetime
import threading
import time

import alpaca_trade_api as tradeapi
import gym
import numpy as np
import pandas as pd
import torch

from meta.data_processors.alpaca import Alpaca


class AlpacaPaperTrading_erl:
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
        latency=None,
    ):
        if agent == "ppo":
            from elegantrl.agent import AgentPPO
            from elegantrl.run import Arguments, init_agent

            config = {"state_dim": state_dim, "action_dim": action_dim}
            args = Arguments(agent=AgentPPO, env=StockEnvEmpty(config))
            args.cwd = cwd
            args.net_dim = net_dim
            try:
                agent = init_agent(args, gpu_id=0)
                self.act = agent.act
                self.device = agent.device
            except Exception as e:
                raise RuntimeError(f"Failed to load agent: {e}")
        else:
            raise ValueError("Agent input is NOT supported yet.")

        try:
            self.alpaca = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, "v2")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Alpaca API: {e}")

        if time_interval == "1s":
            self.time_interval = 1
        elif time_interval == "5s":
            self.time_interval = 5
        elif time_interval == "1Min":
            self.time_interval = 60
        elif time_interval == "5Min":
            self.time_interval = 300
        elif time_interval == "15Min":
            self.time_interval = 900
        else:
            raise ValueError("Unsupported time interval.")

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

    def test_latency(self, test_times=10):
        total_time = 0
        for _ in range(test_times):
            try:
                time0 = time.time()
                self.get_state()
                total_time += time.time() - time0
            except Exception as e:
                print(f"[Latency Test Error] {e}")
        latency = total_time / test_times
        print(f"Latency for data processing: {latency:.4f} seconds")
        return latency

    def run(self):
        try:
            orders = self.alpaca.list_orders(status="open")
            for order in orders:
                self.alpaca.cancel_order(order.id)
        except Exception as e:
            print(f"[Order Cancel Error] {e}")

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
                self.timeToClose = 60  # fallback

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
            with torch.no_grad():
                s_tensor = torch.as_tensor((state,), device=self.device)
                a_tensor = self.act(s_tensor)
                action = a_tensor.detach().cpu().numpy()[0]
        except Exception as e:
            print(f"[Action Error] {e}")
            return

        action = np.nan_to_num(action * self.max_stock).astype(int)
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
                    continue  # skip unknown symbol

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


class StockEnvEmpty(gym.Env):
    def __init__(self, config):
        state_dim = config["state_dim"]
        action_dim = config["action_dim"]
        self.env_num = 1
        self.max_step = 10000
        self.env_name = "StockEnvEmpty"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.if_discrete = False
        self.target_return = 9999
        self.observation_space = gym.spaces.Box(low=-3000, high=3000, shape=(state_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)

    def reset(self):
        return np.zeros(self.state_dim, dtype=np.float32)

    def step(self, actions):
        return np.zeros(self.state_dim, dtype=np.float32), 0.0, True, {}
