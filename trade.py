# two modes: "backtesting" and "paper_trading"
from meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading
from test import test
import logging

logger = logging.getLogger("finrl_trader")
logger.setLevel(logging.INFO)


def trade(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    API_KEY,
    API_SECRET,
    API_BASE_URL,
    trade_mode="backtesting",
    if_vix=True,
    **kwargs
):
    try:
        if trade_mode == "backtesting":
            logger.info("Running in backtesting mode...")
            return test(
                drl_lib=drl_lib,
                env=env,
                model_name=model_name,
                start_date=start_date,
                end_date=end_date,
                ticker_list=ticker_list,
                data_source=data_source,
                time_interval=time_interval,
                technical_indicator_list=technical_indicator_list,
                if_vix=if_vix,
                **kwargs,
            )

        elif trade_mode == "paper_trading":
            logger.info("Running in paper trading mode...")

            try:
                net_dim = kwargs["net_dimension"]
                cwd = kwargs.get("cwd", f"./{model_name}")
                state_dim = kwargs["state_dim"]
                action_dim = kwargs["action_dim"]
            except KeyError as e:
                raise ValueError(
                    f"[trade] Missing required parameter for paper trading: {e}"
                )

            trader = AlpacaPaperTrading(
                ticker_list=ticker_list,
                time_interval=time_interval,
                drl_lib=drl_lib,
                model_name=model_name,
                cwd=cwd,
                net_dim=net_dim,
                state_dim=state_dim,
                action_dim=action_dim,
                API_KEY=API_KEY,
                API_SECRET=API_SECRET,
                API_BASE_URL=API_BASE_URL,
                tech_indicator_list=technical_indicator_list,
                turbulence_thresh=30,
                max_stock=1e2,
                latency=None,
            )
            trader.run()

        else:
            raise ValueError("Invalid mode. Choose either 'backtesting' or 'paper_trading'.")

    except Exception as e:
        logger.error(f"[trade] Failed to execute trading mode '{trade_mode}': {e}")
        return None
