import os
import sys

# Directory Constants
DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

# Date Ranges
TRAIN_START_DATE = "2014-01-01"
TRAIN_END_DATE = "2020-07-31"

TEST_START_DATE = "2020-08-01"
TEST_END_DATE = "2021-10-01"

TRADE_START_DATE = "2021-11-01"
TRADE_END_DATE = "2021-12-01"

# Validate Dates
for date_name, value in {
    "TRAIN_START_DATE": TRAIN_START_DATE,
    "TRAIN_END_DATE": TRAIN_END_DATE,
    "TEST_START_DATE": TEST_START_DATE,
    "TEST_END_DATE": TEST_END_DATE,
    "TRADE_START_DATE": TRADE_START_DATE,
    "TRADE_END_DATE": TRADE_END_DATE,
}.items():
    if not isinstance(value, str) or len(value) != 10:
        print(f"[WARNING] {date_name} is not properly formatted: {value}")

# Indicators (must match stockstats naming)
INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

# Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {
    "batch_size": 100,
    "buffer_size": 1000000,
    "learning_rate": 0.001,
}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}

# Timezones
TIME_ZONE_SHANGHAI = "Asia/Shanghai"
TIME_ZONE_USEASTERN = "US/Eastern"
TIME_ZONE_PARIS = "Europe/Paris"
TIME_ZONE_BERLIN = "Europe/Berlin"
TIME_ZONE_JAKARTA = "Asia/Jakarta"
TIME_ZONE_SELFDEFINED = "xxx"
USE_TIME_ZONE_SELFDEFINED = 0  # 0 = use default, 1 = use custom

# Timezone check
try:
    import pytz

    if USE_TIME_ZONE_SELFDEFINED:
        if TIME_ZONE_SELFDEFINED not in pytz.all_timezones:
            print(f"[WARNING] Custom time zone '{TIME_ZONE_SELFDEFINED}' is invalid.")
except ImportError:
    print("[WARNING] pytz module not found. Time zone validation skipped.")

# API Keys (fallback placeholders)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "xxx")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "xxx")
ALPACA_API_BASE_URL = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")

BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://data.binance.vision/")
