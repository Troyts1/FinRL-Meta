import os
import sys
from argparse import ArgumentParser
from typing import List

# Safely import all config elements
try:
    from meta.config import (
        ALPACA_API_BASE_URL,
        ALPACA_API_KEY,
        ALPACA_API_SECRET,
        DATA_SAVE_DIR,
        ERL_PARAMS,
        INDICATORS,
        RESULTS_DIR,
        RLlib_PARAMS,
        SAC_PARAMS,
        TENSORBOARD_LOG_DIR,
        TEST_END_DATE,
        TEST_START_DATE,
        TRADE_END_DATE,
        TRADE_START_DATE,
        TRAIN_END_DATE,
        TRAIN_START_DATE,
        TRAINED_MODEL_DIR,
    )
    from meta.config_tickers import DOW_30_TICKER
except ImportError as e:
    print(f"[main.py] Config import failed: {e}")
    sys.exit(1)

def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode: train, download_data, backtest",
        metavar="MODE",
        default="train",
    )
    return parser

def check_and_make_directories(directories: List[str]):
    for directory in directories:
        try:
            path = os.path.join(".", directory)
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"[main.py] Failed to create directory {directory}: {e}")

def main():
    try:
        parser = build_parser()
        options = parser.parse_args()

        check_and_make_directories([
            DATA_SAVE_DIR,
            TRAINED_MODEL_DIR,
            TENSORBOARD_LOG_DIR,
            RESULTS_DIR,
        ])

        print(f"[main.py] Mode selected: {options.mode}")

    except Exception as e:
        print(f"[main.py] Unexpected error in main(): {e}")
        sys.exit(1)

## Example usage:
# python main.py --mode=train
# python main.py --mode=test
# python main.py --mode=trade
if __name__ == "__main__":
    main()
