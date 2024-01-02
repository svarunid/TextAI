import configparser
from os import path
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.tree_util import tree_map
from spax.nn.utils import optim

import wandb
from helpers import spickle, text
from models.transformer import transformer

# Loading transformer config
config_dir = (
    Path(path.dirname(path.realpath(__file__))).parent
    / "config"
    / "transformer_config.ini"
)
config = configparser.ConfigParser()
config.read(config_dir)

# Initialising data iterator
dataloader = spickle.sload(config.get("training", "data_path"))
dataloader = text.seq2seq_batched_iterator(
    dataloader,
    config.getint("model", "in_seq_len"),
    config.getint("model", "out_seq_len"),
    config.getint("model", "batch_size"),
)

# Initialising transformer model
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
optimizer = optax.adam(learning_rate=lr)

opt_state, step = optim(
    model, optimizer, loss, in_axes=(None, 0, 0, 0, 0, 0), out_axes=0
)

model_path = path.join(config.get("training", "checkpoint_path"), "model.eqx")
opt_state_path = path.join(config.get("training", "checkpoint_path"), "opt_state.eqx")
if path.isfile(model_path) and path.isfile(opt_state_path):
    model = eqx.tree_deserialise_leaves(model_path, model)
    opt_state = eqx.tree_deserialise_leaves(opt_state_path, opt_state)

if config.getboolean("training", "wandb"):
    wandb_config = dict(config["model"])
    wandb_config["epochs"] = config.getint("training", "epochs")
    wandb_config["dataset"] = config.get("training", "dataset")
    wandb_config["model_parameters"] = sum(
        [x.size for x in jax.tree_util.tree_leaves(model)]
    )

    run = wandb.init(
        project=config.get("training", "wandb_project"),
        notes=config.get("training", "wandb_notes"),
        config=wandb_config,
    )

checkpoint_freq = config.getint("training", "checkpoint_freq")
use_wandb = config.getboolean("training", "wandb")

print("Running...")
for e in range(config.getint("training", "epochs")):
    total_loss = 0
    num_batches = 0
    for Xbt, ybt, labelbt in dataloader:
        Xbt, ybt, labelbt = [jnp.array(x) for x in (Xbt, ybt, labelbt)]
        Xmask, ymask = [text.create_pad_masks(x) for x in (Xbt, ybt)]
        ymask = ymask[:, jnp.newaxis, :] + text.subsequent_mask(ymask.shape[-1])

        model, opt_state, batch_loss = step(
            model, opt_state, Xbt, ybt, Xmask, ymask, labelbt
        )
        total_loss += batch_loss
        num_batches += 1

        if num_batches % checkpoint_freq == 0:
            eqx.tree_serialise_leaves(model_path, model)
            eqx.tree_deserialise_leaves(opt_state_path, opt_state)
            print(
                f"Batches trained: {num_batches} | Avg. Batch loss: {total_loss/num_batches}"
            )

        if use_wandb:
            wandb.log({"loss": total_loss / num_batches})

    epoch_loss = total_loss / num_batches
    print(f"Epoch {e} | loss: {epoch_loss}")
