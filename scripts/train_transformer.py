import configparser
import os
import pickle

import jax
import optax
import wandb
from etils import epath
from jax import config as jax_config
from jax import numpy as jnp
from orbax import checkpoint as ocp
from spax.nn.utils import optim

from helpers import spickle, text
from models.transformer import transformer

jax_config.update("jax_debug_nans", True)
jax_config.update("jax_debug_infs", True)

# Loading transformer config
config_dir = (
    epath.Path(os.path.dirname(os.path.realpath(__file__))).parent
    / "config"
    / "transformer.ini"
)
config = configparser.ConfigParser()
config.read(config_dir)

# Initialize data iterator
dataloader = spickle.sload(config.get("training", "data_path"))
dataloader = text.seq2seq_batched_iterator(
    dataloader,
    config.getint("model", "in_seq_len"),
    config.getint("model", "out_seq_len"),
    config.getint("model", "batch_size"),
)

# load validation data
if use_validation := config.getboolean("validation", "use_validation"):
    val_data = list(pickle.load(open(config.get("validation", "data_path"), "rb")))
    val_data = text.seq2seq_batched_iterator(
        val_data,
        config.getint("model", "in_seq_len"),
        config.getint("model", "out_seq_len"),
        len(val_data),
    )

# Initialize transformer model
model, forward = transformer(
    jax.random.PRNGKey(config.getint("model", "seed")),
    config.getint("model", "in_vocab"),
    config.getint("model", "out_vocab"),
    config.getint("model", "in_seq_len"),
    config.getint("model", "out_seq_len"),
    config.getint("model", "d_model"),
    config.getint("model", "num_heads"),
    config.getint("model", "d_ff"),
    config.getint("model", "enc_layers"),
    config.getint("model", "dec_layers"),
)


# Defining loss function
# Accepts padded lables too
@jax.jit
def loss(model, X, y, X_mask, y_mask, labels):
    y_pred = jnp.log(forward(model, X, y, X_mask, y_mask))
    y_pred = jnp.where(labels == 0, 0, jnp.take(y_pred, labels, axis=-1))
    count = jnp.count_nonzero(y_pred)
    return -jnp.sum(y_pred) / count


# Defining optimiser
# A linear warmup is used for the first 200 steps upto a peak learning rate of 0.1
# The learning rate is then decayed using a cosine decay schedule for 2000 steps
lr = config.getfloat("training", "lr")
lr = optax.warmup_cosine_decay_schedule(lr, 0.1, 200, 2000, 0.0001)
optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lr)
del lr

opt_state, step = optim(
    model, optimizer, loss, in_axes=(None, 0, 0, 0, 0, 0), out_axes=0
)

# Make checkpoint directory if it doesn't exist and initialize checkpoint manager
if use_checkpoint := config.getboolean("checkpoint", "use_checkpoint"):
    checkpoint_path = epath.Path(config.get("checkpoint", "path"))
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    freq = config.getint("checkpoint", "freq")
    options = ocp.CheckpointManagerOptions(
        max_to_keep=config.getint("checkpoint", "max_to_keep"),
        save_interval_steps=freq,
    )
    mngr = ocp.CheckpointManager(
        checkpoint_path.resolve(),
        {
            "model": ocp.PyTreeCheckpointer(),
            "opt_state": ocp.PyTreeCheckpointer(),
        },
        options,
    )
    del checkpoint_path, options

    # Loading model and optimiser state if they exist
    if mngr.latest_step() is not None:
        mngr.wait_until_finished()
        restored_items = mngr.restore(mngr.latest_step())
        model = restored_items["model"]
        opt_state = restored_items["opt_state"]

# Optional configuration for logging to wandb
if use_wandb := config.getboolean("wandb", "use_wandb"):
    wandb_config = dict(config["model"])
    wandb_config["epochs"] = config.getint("training", "epochs")
    wandb_config["dataset_name"] = config.get("training", "dataset_name")
    wandb_config["model_parameters"] = sum(
        [x.size for x in jax.tree_util.tree_leaves(model)]
    )

    run = wandb.init(
        project=config.get("wandb", "project"),
        notes=config.get("wandb", "notes"),
        name=config.get("wandb", "name"),
        config=wandb_config,
    )
    del wandb_config

# Process validation data
if use_validation:
    Xdev, ydev, labeldev = next(val_data)
    Xdev, ydev, labeldev = [jnp.array(x) for x in (Xdev, ydev, labeldev)]
    Xdev_mask, ydev_mask = [text.create_pad_masks(x) for x in (Xdev, ydev)]
    ydev_mask = ydev_mask[:, jnp.newaxis, :] + text.subsequent_mask(ydev_mask.shape[-1])
    vmapped_loss = jax.vmap(loss, in_axes=(None, 0, 0, 0, 0, 0), out_axes=0)
    del val_data # Delete validation data to free up memory

# Load train loop config
batches_trained = config.getint("training", "batches_trained")
epochs_trained = config.getint("training", "epochs_trained")

# Training loop
print("Running...")
for e in range(config.getint("training", "epochs")):
    # Skip epochs that have already been trained
    if e >= epochs_trained:
        total_loss = 0
        num_batches = 0
        for i, (Xbt, ybt, labelbt) in enumerate(dataloader):
            # Skip batches that have already been trained
            if i >= batches_trained:
                Xbt, ybt, labelbt = [jnp.array(x) for x in (Xbt, ybt, labelbt)]
                Xmask, ymask = [text.create_pad_masks(x) for x in (Xbt, ybt)]
                ymask = ymask[:, jnp.newaxis, :] + text.subsequent_mask(ymask.shape[-1])

                model, opt_state, batch_loss = step(
                    model, opt_state, Xbt, ybt, Xmask, ymask, labelbt
                )
                total_loss += batch_loss
                num_batches += 1
                print(
                    f"Batches trained: {num_batches} | Avg. Batch loss: {total_loss/num_batches}"
                )

                # Checkpoint model and optimiser state
                if use_checkpoint:
                    print(f"Step: {i * (e + 1)}")
                    print(f"Should save: {mngr.should_save(i * (e + 1))}")
                    mngr.save(
                        i * (e + 1),
                        {
                            "model": model,
                            "opt_state": opt_state,
                        }
                    )

                # Log to wandb
                if use_wandb:
                    log = {
                        "training_loss": total_loss / num_batches,
                        "learning_rate": optimizer["learning_rate"],
                    }
                    if use_validation:
                        log["validation_loss"] = vmapped_loss(
                            model, Xdev, ydev, Xdev_mask, ydev_mask, labeldev
                        )
                    wandb.log(log)
                config.set("training", "batches_trained", str(num_batches))

        config.set("training", "epochs_trained", str(e + 1))
        epoch_loss = total_loss / num_batches
        print(f"Epoch {e} | loss: {epoch_loss}")
