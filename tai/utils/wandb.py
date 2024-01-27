import wandb
from clu.periodic_actions import PeriodicCallback


def _periodic_log(steps):
    """
    Create a periodic callback that logs to wandb.

    Args:
        steps: int
            Number of steps between each call to the callback.

    Returns:
        A PeriodicCallback object.
    """
    pc = PeriodicCallback(
        steps,
        callback_fn=wandb.log,
        execute_async=True,
        pass_step_and_time=False,
    )
    return pc


def configure_wandb(config, model_config):
    """
    Configure wandb from a config dictionary.

    Args:
        config: dict
            A dictionary containing the configuration for wandb.
        model_config: dict
            A dictionary containing the configuration for the model.

    Returns:
        A wandb.Run object.
    """
    run_config = config["run_config"]
    run_config["epochs"] = config["epochs"]
    run_config["model_parameters"] = config["params"]
    run_config = dict(**run_config, **model_config)

    wandb.init(
        project=config["project"],
        notes=config["notes"],
        name=config["name"],
        config=run_config,
    )

    return _periodic_log(config["freq"])
