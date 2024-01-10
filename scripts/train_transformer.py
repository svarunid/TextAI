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

# TODO: Use common loop utils.

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
    Xdev, ydev, labeldev = next(val_data)
    Xdev, ydev, labeldev = [jnp.array(x) for x in (Xdev, ydev, labeldev)]
    Xdev_mask, ydev_mask = [text.create_pad_masks(x) for x in (Xdev, ydev)]
    ydev_mask = ydev_mask[:, jnp.newaxis, :] + text.subsequent_mask(ydev_mask.shape[-1])
    del val_data  # Delete validation data to free up memory

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
# Expects labels to be padded
@jax.jit
def loss(model, X, y, Xmask, ymask, labels):
    yhat = forward(model, X, y, Xmask, ymask)
    yhat = jax.nn.log_softmax(yhat, axis=-1)
    yhat = jnp.where(labels == 0, 0, jnp.take(yhat, labels, axis=-1))
    count = jnp.count_nonzero(yhat)
    return -jnp.sum(yhat) / count

if use_validation:
    vmapped_loss = jax.vmap(loss, in_axes=(None, 0, 0, 0, 0, 0), out_axes=0)

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
batches_trained = 0
epochs_trained = 0
if use_checkpoint := config.getboolean("checkpoint", "use_checkpoint"):
    freq = config.getint("checkpoint", "freq")
    checkpoint_path = epath.Path(config.get("checkpoint", "path"))
    if checkpoint_path.exists() and config.getboolean("checkpoint", "overwrite"):
        checkpoint_path.rmtree()
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=config.getint("checkpoint", "max_to_keep"),
        save_interval_steps=freq,
    )
    mngr = ocp.CheckpointManager(
        checkpoint_path.resolve(),
        {
            "model": ocp.AsyncCheckpointer(
                ocp.PyTreeCheckpointHandler(ocdbt_merge=False)
            ),
            "opt_state": ocp.AsyncCheckpointer(
                ocp.PyTreeCheckpointHandler(ocdbt_merge=False)
            ),
        },
        options,
    )
    del checkpoint_path, options

    # Loading model, optimiser state and training loop config if checkpoint exists
    if mngr.latest_step() is not None:
        mngr.wait_until_finished()
        restored_items = mngr.restore(mngr.latest_step())
        model = restored_items["model"]
        opt_state = restored_items["opt_state"]
        batches_trained = mngr.latest_step()
        if num_batches := config.getint("training", "num_batches", fallback=0):
            batches_trained = batches_trained % num_batches
            epochs_trained = batches_trained // num_batches

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
                num_batches = i + 1

                # Checkpoint model and optimiser state
                if use_checkpoint:
                    if i % freq == 0:
                        mngr.save(
                            i * (e + 1),
                            {
                                "model": model,
                                "opt_state": opt_state,
                            },
                        )

                # Log to wandb
                if use_wandb:
                    log = {
                        "training_loss": total_loss / num_batches,
                        "learning_rate": opt_state.hyperparams["learning_rate"],
                    }
                    if use_validation:
                        log["validation_loss"] = vmapped_loss(
                            model, Xdev, ydev, Xdev_mask, ydev_mask, labeldev
                        )
                    wandb.log(log)

        if e == 0:
            with open(config_dir, "w") as f:
                config.set("training", "num_batches", str(num_batches))
                config.write(f)
        epoch_loss = total_loss / num_batches
        print(f"Epoch {e} | loss: {epoch_loss}")
