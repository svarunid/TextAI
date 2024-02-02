from clu.periodic_actions import PeriodicCallback
from orbax import checkpoint as ocp

from tai.utils import data


def create_checkpoint_manager(root_dir, config):
    """
    Create a checkpoint manager from a config dictionary.
    """
    checkpoint_path = root_dir / config["path"]
    if checkpoint_path.exists() and config["overwrite"]:
        checkpoint_path.rmtree()
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    options = ocp.CheckpointManagerOptions(
        max_to_keep=config["max_to_keep"],
        save_interval_steps=config["freq"],
    )

    mngr = ocp.CheckpointManager(
        checkpoint_path.resolve(),
        options=options,
    )

    return mngr


class PeriodicCheckpoint:
    """
    A wrapper that combines a checkpoint manager and a dataset iterator
    to save periodically.
    """

    def __init__(self, root_dir, config, ds):
        """
        Initializes the checkpoint manager.

        Args:
            root_dir (pathlib.Path): Root directory.
            config (dict): Configuration dictionary.
            ds (tf.data.Dataset): Dataset.
        """
        self.ckpt_manager = create_checkpoint_manager(root_dir, config)
        self.dataloader = data.TfDatasetIterator(ds, config["path"])
        self.periodic_save = PeriodicCallback(
            config["freq"],
            callback_fn=lambda step, args: (
                self.ckpt_manager.save(step, args),
                self.dataloader.save("dataset"),
            ),
            execute_async=True,
            pass_step_and_time=False,
        )

    def restore(self, step):
        """
        Restore the checkpoint manager and the dataset.

        Args:
            step (int): Step to restore from.
        """
        return (self.ckpt_manager.restore(step), self.dataloader.restore("dataset"))

    def save(self, step, args):
        """
        Save the checkpoint manager and the dataset.

        Args:
            step (int): Step to save at.
            args (clu.checkpoint.SaveArgs): Arguments to save.
        """
        self.periodic_save(step=step, args=args)
        self.dataloader.save("dataset")

    def wait_until_finished(self):
        """
        Wait until the checkpoint manager has finished saving.
        """
        self.ckpt_manager.wait_until_finished()

    def latest_step(self):
        """
        Return the latest step saved.
        """
        return self.ckpt_manager.latest_step()
