# use three drl libraries: elegantrl, rllib, and SB3
from agents.elegantrl_models import DRLAgent as DRLAgent_erl
from agents.rllib_models import DRLAgent as DRLAgent_rllib
from agents.stablebaselines3_models import DRLAgent as DRLAgent_sb3

import logging
logger = logging.getLogger("finrl_trader")
logger.setLevel(logging.INFO)

def test(drl_lib, env, model_name, **kwargs):
    """
    Load a trained model and run prediction with the specified DRL library.
    
    :param drl_lib: str, one of ["elegantrl", "rllib", "stable_baselines3"]
    :param env: gym-compatible trading environment
    :param model_name: str, name of the trained model
    :param kwargs: optional parameters like net_dimension or cwd
    :return: episode total asset list or prediction output
    """
    net_dimension = kwargs.get("net_dimension", 2**7)
    cwd = kwargs.get("cwd", f"./{model_name}")

    try:
        if drl_lib == "elegantrl":
            return DRLAgent_erl.DRL_prediction(
                model_name=model_name,
                cwd=cwd,
                net_dimension=net_dimension,
                environment=env,
            )

        elif drl_lib == "rllib":
            return DRLAgent_rllib.DRL_prediction(
                model_name=model_name,
                env=env,
                agent_path=cwd,
            )

        elif drl_lib == "stable_baselines3":
            return DRLAgent_sb3.DRL_prediction_load_from_file(
                model_name=model_name,
                environment=env,
                cwd=cwd,
            )

        else:
            raise ValueError(f"Unsupported DRL library: {drl_lib}. Choose from 'elegantrl', 'rllib', 'stable_baselines3'.")

    except Exception as e:
        logger.error(f"[test] Prediction failed using {drl_lib} with model {model_name}: {e}")
        return None
