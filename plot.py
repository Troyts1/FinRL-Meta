from copy import deepcopy
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfolio
from pyfolio import timeseries
from datetime import datetime

from finrl.config import TRAIN_START_DATE
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from meta import config

import warnings
warnings.filterwarnings("ignore")


def get_daily_return(df, value_col_name="account_value"):
    try:
        df = deepcopy(df)
        df["daily_return"] = df[value_col_name].pct_change(1)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date", "daily_return"], inplace=True)
        df.set_index("date", inplace=True, drop=True)
        df.index = df.index.tz_localize("UTC")
        return pd.Series(df["daily_return"], index=df.index)
    except Exception as e:
        print(f"[get_daily_return] Error: {e}")
        return pd.Series(dtype=float)


def convert_daily_return_to_pyfolio_ts(df):
    try:
        strategy_ret = df.copy()
        strategy_ret["date"] = pd.to_datetime(strategy_ret["date"], errors="coerce")
        strategy_ret.dropna(subset=["date", "daily_return"], inplace=True)
        strategy_ret.set_index("date", drop=False, inplace=True)
        strategy_ret.index = strategy_ret.index.tz_localize("UTC")
        return pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)
    except Exception as e:
        print(f"[convert_daily_return_to_pyfolio_ts] Error: {e}")
        return pd.Series(dtype=float)


def backtest_stats(account_value, value_col_name="account_value"):
    try:
        dr_test = get_daily_return(account_value, value_col_name=value_col_name)
        perf_stats_all = timeseries.perf_stats(
            returns=dr_test,
            positions=None,
            transactions=None,
            turnover_denom="AGB",
        )
        print(perf_stats_all)
        return perf_stats_all
    except Exception as e:
        print(f"[backtest_stats] Error: {e}")
        return None


def backtest_plot(account_value,
                  baseline_start=config.TRADE_START_DATE,
                  baseline_end=config.TRADE_END_DATE,
                  baseline_ticker="^DJI",
                  value_col_name="account_value"):
    try:
        df = deepcopy(account_value)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)
        test_returns = get_daily_return(df, value_col_name=value_col_name)

        baseline_df = get_baseline(ticker=baseline_ticker, start=baseline_start, end=baseline_end)
        baseline_df["date"] = pd.to_datetime(baseline_df["date"], format="%Y-%m-%d", errors="coerce")
        baseline_df.dropna(subset=["date"], inplace=True)
        baseline_df = pd.merge(df[["date"]], baseline_df, how="left", on="date")
        baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
        baseline_returns = get_daily_return(baseline_df, value_col_name="close")

        with pyfolio.plotting.plotting_context(font_scale=1.1):
            pyfolio.create_full_tear_sheet(
                returns=test_returns,
                benchmark_rets=baseline_returns,
                set_context=False,
            )
    except Exception as e:
        print(f"[backtest_plot] Error: {e}")


def get_baseline(ticker, start, end):
    try:
        return YahooDownloader(start_date=start, end_date=end, ticker_list=[ticker]).fetch_data()
    except Exception as e:
        print(f"[get_baseline] Error fetching baseline {ticker}: {e}")
        return pd.DataFrame(columns=["date", "close"])


def trx_plot(df_trade, df_actions, ticker_list):
    try:
        df_trx = pd.DataFrame(np.array(df_actions["transactions"].to_list()))
        df_trx.columns = ticker_list
        df_trx.index = pd.to_datetime(df_actions["date"], errors="coerce")
        df_trx.index.name = ""

        for i in range(df_trx.shape[1]):
            try:
                df_trx_temp = df_trx.iloc[:, i]
                df_trx_temp_sign = np.sign(df_trx_temp)
                buying_signal = df_trx_temp_sign.apply(lambda x: x > 0)
                selling_signal = df_trx_temp_sign.apply(lambda x: x < 0)

                tic_plot = df_trade[
                    (df_trade["tic"] == df_trx_temp.name)
                    & (df_trade["date"].isin(df_trx.index))
                ]["close"]

                tic_plot.index = df_trx_temp.index
                plt.figure(figsize=(10, 8))
                plt.plot(tic_plot, color="g", lw=2.0)
                plt.plot(tic_plot, "^", markersize=10, color="m", label="buy", markevery=buying_signal)
                plt.plot(tic_plot, "v", markersize=10, color="k", label="sell", markevery=selling_signal)
                plt.title(f"{df_trx_temp.name} Transactions: {buying_signal.sum() + selling_signal.sum()}")
                plt.legend()
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=25))
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"[trx_plot] Failed to plot for {df_trx.columns[i]}: {e}")
    except Exception as e:
        print(f"[trx_plot] Error generating transaction plot: {e}")
