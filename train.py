from agents.elegantrl_models import DRLAgent as DRLAgent_erl
from agents.rllib_models import DRLAgent as DRLAgent_rllib
from agents.stablebaselines3_models import DRLAgent as DRLAgent_sb3
import logging

logger = logging.getLogger("finrl_trader")
logger.setLevel(logging.INFO)

def train(drl_lib, env_train, model_name, **kwargs):
    cwd = kwargs.get("cwd", f"./{model_name}")

    try:
        if drl_lib == "elegantrl":
            break_step = kwargs.get("break_step", 1e6)
            erl_params = kwargs.get("erl_params", {})

            agent = DRLAgent_erl(env=env_train)
            model = agent.get_model(model_name, model_kwargs=erl_params)

            trained_model = agent.train_model(
                model=model,
                cwd=cwd,
                total_timesteps=break_step
            )
            return trained_model

        elif drl_lib == "rllib":
            total_episodes = kwargs.get("total_episodes", 100)
            rllib_params = kwargs.get("rllib_params", {})

            required_keys = ["lr", "train_batch_size", "gamma"]
            for key in required_keys:
                if key not in rllib_params:
                    raise ValueError(f"[train][rllib] Missing required rllib_param: {key}")

            agent_rllib = DRLAgent_rllib(env=env_train)
            model, model_config = agent_rllib.get_model(model_name)

            model_config["lr"] = rllib_params["lr"]
            model_config["train_batch_size"] = rllib_params["train_batch_size"]
            model_config["gamma"] = rllib_params["gamma"]

            trained_model = agent_rllib.train_model(
                model=model,
                model_name=model_name,
                model_config=model_config,
                total_episodes=total_episodes,
            )
            trained_model.save(cwd)
            return trained_model

        elif drl_lib == "stable_baselines3":
            total_timesteps = kwargs.get("total_timesteps", 1e6)
            agent_params = kwargs.get("agent_params", {})

            agent = DRLAgent_sb3(env=env_train)
            model = agent.get_model(model_name, model_kwargs=agent_params)

            trained_model = agent.train_model(
                model=model,
                tb_log_name=model_name,
                total_timesteps=total_timesteps
            )
            logger.info("Training finished!")
            trained_model.save(cwd)
            logger.info(f"Trained model saved in {cwd}")
            return trained_model

        else:
            raise ValueError(f"[train] Unsupported DRL library: {drl_lib}")

    except Exception as e:
        logger.error(f"[train] Training failed for {drl_lib} using model {model_name}: {e}")
        return None
