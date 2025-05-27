import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from meta.config_tickers import DOW_30_TICKER
from meta.data_processors._base import DataSource

class DataProcessor:
    def __init__(self, data_source: DataSource, start_date: str, end_date: str, time_interval: str, **kwargs):
        self.data_source = data_source
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval
        self.dataframe = pd.DataFrame()

        processor_dict = {}
        try:
            if self.data_source == DataSource.akshare:
                from meta.data_processors.akshare import Akshare
                processor_dict[self.data_source] = Akshare
            elif self.data_source == DataSource.alpaca:
                from meta.data_processors.alpaca import Alpaca
                processor_dict[self.data_source] = Alpaca
            elif self.data_source == DataSource.alphavantage:
                from meta.data_processors.alphavantage import Alphavantage
                processor_dict[self.data_source] = Alphavantage
            elif self.data_source == DataSource.baostock:
                from meta.data_processors.baostock import Baostock
                processor_dict[self.data_source] = Baostock
            elif self.data_source == DataSource.binance:
                from meta.data_processors.binance import Binance
                processor_dict[self.data_source] = Binance
            elif self.data_source == DataSource.ccxt:
                from meta.data_processors.ccxt import Ccxt
                processor_dict[self.data_source] = Ccxt
            elif self.data_source == DataSource.iexcloud:
                from meta.data_processors.iexcloud import Iexcloud
                processor_dict[self.data_source] = Iexcloud
            elif self.data_source == DataSource.joinquant:
                from meta.data_processors.joinquant import Joinquant
                processor_dict[self.data_source] = Joinquant
            elif self.data_source == DataSource.quandl:
                from meta.data_processors.quandl import Quandl
                processor_dict[self.data_source] = Quandl
            elif self.data_source == DataSource.quantconnect:
                from meta.data_processors.quantconnect import Quantconnect
                processor_dict[self.data_source] = Quantconnect
            elif self.data_source == DataSource.ricequant:
                from meta.data_processors.ricequant import Ricequant
                processor_dict[self.data_source] = Ricequant
            elif self.data_source == DataSource.tushare:
                from meta.data_processors.tushare import Tushare
                processor_dict[self.data_source] = Tushare
            elif self.data_source == DataSource.wrds:
                from meta.data_processors.wrds import Wrds
                processor_dict[self.data_source] = Wrds
            elif self.data_source == DataSource.yahoofinance:
                from meta.data_processors.yahoofinance import Yahoofinance
                processor_dict[self.data_source] = Yahoofinance
            else:
                print(f"[WARNING] {self.data_source} is NOT supported yet.")
        except ImportError as e:
            raise ImportError(f"[ERROR] Failed to import processor: {e}")

        try:
            self.processor = processor_dict.get(self.data_source)(
                data_source, start_date, end_date, time_interval, **kwargs
            )
            print(f"[INFO] {self.data_source} successfully connected")
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to initialize {self.data_source}: {e}")

    def download_data(self, ticker_list):
        try:
            self.processor.download_data(ticker_list=ticker_list)
            self.dataframe = self.processor.dataframe
        except Exception as e:
            print(f"[ERROR] download_data failed: {e}")
            self.dataframe = pd.DataFrame()

    def clean_data(self):
        try:
            self.processor.dataframe = self.dataframe
            self.processor.clean_data()
            self.dataframe = self.processor.dataframe
        except Exception as e:
            print(f"[ERROR] clean_data failed: {e}")

    def add_technical_indicator(self, tech_indicator_list: List[str], select_stockstats_talib: int = 0):
        try:
            self.tech_indicator_list = tech_indicator_list
            self.processor.add_technical_indicator(tech_indicator_list, select_stockstats_talib)
            self.dataframe = self.processor.dataframe
        except Exception as e:
            print(f"[ERROR] add_technical_indicator failed: {e}")

    def add_turbulence(self):
        try:
            self.processor.add_turbulence()
            self.dataframe = self.processor.dataframe
        except Exception as e:
            print(f"[ERROR] add_turbulence failed: {e}")

    def add_vix(self):
        try:
            self.processor.add_vix()
            self.dataframe = self.processor.dataframe
        except Exception as e:
            print(f"[ERROR] add_vix failed: {e}")

    def df_to_array(self, if_vix: bool) -> np.array:
        try:
            price_array, tech_array, turbulence_array = self.processor.df_to_array(
                self.tech_indicator_list, if_vix
            )
            tech_array[np.isnan(tech_array)] = 0
            return price_array, tech_array, turbulence_array
        except Exception as e:
            print(f"[ERROR] df_to_array failed: {e}")
            return np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))

    def data_split(self, df, start, end, target_date_col="time"):
        try:
            data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
            data = data.sort_values([target_date_col, "tic"], ignore_index=True)
            data.index = data[target_date_col].factorize()[0]
            return data
        except Exception as e:
            print(f"[ERROR] data_split failed: {e}")
            return pd.DataFrame()

    def fillna(self):
        try:
            self.processor.dataframe = self.dataframe
            self.processor.fillna()
            self.dataframe = self.processor.dataframe
        except Exception as e:
            print(f"[ERROR] fillna failed: {e}")

    def run(self, ticker_list: str, technical_indicator_list: List[str], if_vix: bool, cache: bool = False, select_stockstats_talib: int = 0):
        if self.time_interval == "1s" and self.data_source != DataSource.binance:
            raise ValueError("1s interval is only supported with Binance as the data source.")

        cache_filename = "_".join(ticker_list + [self.data_source, self.start_date, self.end_date, self.time_interval]) + ".pickle"
        cache_dir = "./cache"
        cache_path = os.path.join(cache_dir, cache_filename)

        try:
            if cache and os.path.isfile(cache_path):
                print(f"[INFO] Using cached file {cache_path}")
                self.tech_indicator_list = technical_indicator_list
                with open(cache_path, "rb") as handle:
                    self.processor.dataframe = pickle.load(handle)
            else:
                self.download_data(ticker_list)
                self.clean_data()
                if cache:
                    os.makedirs(cache_dir, exist_ok=True)
                    with open(cache_path, "wb") as handle:
                        pickle.dump(self.dataframe, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[ERROR] Cache loading failed: {e}")

        self.add_technical_indicator(technical_indicator_list, select_stockstats_talib)
        if if_vix:
            self.add_vix()

        return self.df_to_array(if_vix)


def check_joinquant():
    try:
        TRADE_START_DATE = "2022-09-01"
        TRADE_END_DATE = "2023-11-01"
        TIME_INTERVAL = "1d"
        TECHNICAL_INDICATOR = [
            "macd", "boll_ub", "boll_lb", "rsi_30", "dx_30", "close_30_sma", "close_60_sma"
        ]
        kwargs = {"username": "xxx", "password": "xxx"}
        p = DataProcessor(
            data_source=DataSource.joinquant,
            start_date=TRADE_START_DATE,
            end_date=TRADE_END_DATE,
            time_interval=TIME_INTERVAL,
            **kwargs,
        )
        ticker_list = ["000612.XSHE", "601808.XSHG"]
        p.download_data(ticker_list)
        p.clean_data()
        p.add_turbulence()
        p.add_technical_indicator(TECHNICAL_INDICATOR)
        p.add_vix()
        p.run(ticker_list, TECHNICAL_INDICATOR, if_vix=False, cache=True)
    except Exception as e:
        print(f"[ERROR] check_joinquant failed: {e}")


def check_yahoofinance():
    try:
        data_source = DataSource.yahoofinance
        TRADE_START_DATE = "2022-09-01"
        TRADE_END_DATE = "2023-11-01"
        TIME_INTERVAL = "1d"
        TECHNICAL_INDICATOR = [
            "macd", "boll_ub", "boll_lb", "rsi_30", "dx_30", "close_30_sma", "close_60_sma"
        ]
        kwargs = {}
        p = DataProcessor(
            data_source=data_source,
            start_date=TRADE_START_DATE,
            end_date=TRADE_END_DATE,
            time_interval=TIME_INTERVAL,
            **kwargs,
        )
        ticker_list = DOW_30_TICKER
        p.download_data(ticker_list)
        p.clean_data()
        p.add_turbulence()
        p.add_technical_indicator(TECHNICAL_INDICATOR)
        p.add_vix()
        p.run(ticker_list, TECHNICAL_INDICATOR, if_vix=False, cache=True)
    except Exception as e:
        print(f"[ERROR] check_yahoofinance failed: {e}")


if __name__ == "__main__":
    check_yahoofinance()
