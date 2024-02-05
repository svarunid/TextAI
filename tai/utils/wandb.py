import wandb
from clu.periodic_actions import PeriodicCallback


def configure_wandb(config, model_config):
    """
    Configure wandb from a config dictionary.

    Args:
        config: dict
            A dictionary containing the configuration for wandb.
        model_config: dict
            A dictionary containing the configuration for the model.

    Returns:
        PeriodicCallback: A periodic callback to log to wandb.
    """

    def periodic_log(steps):
        pc = PeriodicCallback(
            steps,
            callback_fn=wandb.log,
            execute_async=True,
            pass_step_and_time=False,
        )
        return pc

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

    return periodic_log(config["freq"])
