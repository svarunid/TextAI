import os

import flax.linen as nn
import jax
import optax
import tensorflow as tf
import wandb
import yaml
from etils import epath
from jax import config as jax_config
from jax import numpy as jnp
from orbax import checkpoint as ocp

from tai.models.transformer import Transformer, TransformerConfig
from tai.utils.data import create_dataset

# TODO: Use common loop utils.

# Configuring JAX & tensorflow
jax_config.update("jax_debug_nans", True)
jax_config.update("jax_debug_infs", True)
# tf.config.experimental.set_visible_devices([], "GPU")

# Loading configuration
root_dir = epath.Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
config_dir = root_dir / "config" / "transformer.yaml"
with open(config_dir, "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)

model_config = config["model"]
train_config = config["training"]
tok_config = config["tokenizer"]
wandb_config = config["wandb"]
checkpoint_config = config["checkpoint"]

# Initialize dataset
tok_config["path"] = root_dir / tok_config["path"]
train_config["vocab_size"] = model_config["out_vocab_size"]
src = root_dir / train_config["data_path"] / train_config["src"]
tgt = root_dir / train_config["data_path"] / train_config["tgt"]
ds = create_dataset(train_config, tok_config, src, tgt)


# Initialize transformer model
param_key, dropout_key = jax.random.split(jax.random.PRNGKey(config["seed"]))
model_config = TransformerConfig.fromDict(model_config)
model = Transformer(model_config)
params = model.init(
    {
        "params": param_key,
        "dropout": dropout_key,
    },
    jnp.ones((train_config["src_max_len"]), dtype=jnp.int32),
    jnp.ones((train_config["tgt_max_len"]), dtype=jnp.int32),
)

# # Defining optimiser
# # A linear warmup is used for the first 200 steps upto a peak learning rate of 0.1
# # The learning rate is then decayed using a cosine decay schedule for 2000 steps
# lr = config.getfloat("training", "lr")
# lr = optax.warmup_cosine_decay_schedule(lr, 0.1, 200, 2000, 0.0001)
# vmapped_loss = jax.vmap(loss, in_axes=(None, 0, 0, 0, 0, 0), out_axes=0)
# optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lr)
# del lr


# # Defining training step
# def step(model, opt_state, X, y, Xmask, ymask, labels):
#     loss, grads = jax.gvalue_and_grad(vmapped_loss)(model, X, y, Xmask, ymask, labels)
#     loss = jnp.mean(loss)
#     grads = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), grads)
#     updates, opt_state = optimizer.update(grads, opt_state, model)
#     model = optax.apply_updates(model, updates)
#     return model, opt_state, loss


# # Initialize optimiser state
# opt_state = optimizer.init(model)

# # Make checkpoint directory if it doesn't exist and initialize checkpoint manager
# batches_trained = 0
# epochs_trained = 0
# if use_checkpoint := config.getboolean("checkpoint", "use_checkpoint"):
#     freq = config.getint("checkpoint", "freq")
#     checkpoint_path = epath.Path(config.get("checkpoint", "path"))
#     if checkpoint_path.exists() and config.getboolean("checkpoint", "overwrite"):
#         checkpoint_path.rmtree()
#         checkpoint_path.mkdir(parents=True, exist_ok=True)
#     options = ocp.CheckpointManagerOptions(
#         max_to_keep=config.getint("checkpoint", "max_to_keep"),
#         save_interval_steps=freq,
#     )
#     mngr = ocp.CheckpointManager(
#         checkpoint_path.resolve(),
#         {
#             "model": ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
#             "opt_state": ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
#         },
#         options,
#     )
#     del checkpoint_path, options

#     # Loading model, optimiser state and training loop config if checkpoint exists
#     if mngr.latest_step() is not None:
#         mngr.wait_until_finished()
#         restored_items = mngr.restore(mngr.latest_step())
#         model = restored_items["model"]
#         opt_state = restored_items["opt_state"]
#         batches_trained = mngr.latest_step()
#         if num_batches := config.getint("training", "num_batches", fallback=0):
#             batches_trained = batches_trained % num_batches
#             epochs_trained = batches_trained // num_batches

# # Optional configuration for logging to wandb
# if use_wandb := config.getboolean("wandb", "use_wandb"):
#     wandb_config = dict(config["model"])
#     wandb_config["epochs"] = config.getint("training", "epochs")
#     wandb_config["dataset_name"] = config.get("training", "dataset_name")
#     wandb_config["model_parameters"] = sum(
#         [x.size for x in jax.tree_util.tree_leaves(model)]
#     )

#     run = wandb.init(
#         project=config.get("wandb", "project"),
#         notes=config.get("wandb", "notes"),
#         name=config.get("wandb", "name"),
#         config=wandb_config,
#     )
#     del wandb_config

# # Training loop
# print("Running...")
# for e in range(config.getint("training", "epochs")):
#     # Skip epochs that have already been trained
#     if e >= epochs_trained:
#         total_loss = 0
#         num_batches = 0
#         for i, (Xbt, ybt, labelbt) in enumerate(dataloader):
#             # Skip batches that have already been trained
#             if i >= batches_trained:
#                 Xbt, ybt, labelbt = [jnp.array(x) for x in (Xbt, ybt, labelbt)]
#                 Xmask, ymask = [text.create_pad_masks(x) for x in (Xbt, ybt)]
#                 ymask = ymask[:, jnp.newaxis, :] + text.subsequent_mask(ymask.shape[-1])

#                 model, opt_state, batch_loss = step(
#                     model, opt_state, Xbt, ybt, Xmask, ymask, labelbt
#                 )
#                 total_loss += batch_loss
#                 num_batches = i + 1

#                 # Checkpoint model and optimiser state
#                 if use_checkpoint:
#                     if i % freq == 0:
#                         mngr.save(
#                             i * (e + 1),
#                             {
#                                 "model": model,
#                                 "opt_state": opt_state,
#                             },
#                         )

#                 # Log to wandb
#                 if use_wandb:
#                     log = {
#                         "training_loss": total_loss / num_batches,
#                         "learning_rate": opt_state.hyperparams["learning_rate"],
#                     }
#                     if use_validation:
#                         log["validation_loss"] = vmapped_loss(
#                             model, Xdev, ydev, Xdev_mask, ydev_mask, labeldev
#                         )
#                     wandb.log(log)

#         if e == 0:
#             with open(config_dir, "w") as f:
#                 config.set("training", "num_batches", str(num_batches))
#                 config.write(f)
#         epoch_loss = total_loss / num_batches
#         print(f"Epoch {e} | loss: {epoch_loss}")
