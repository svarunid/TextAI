import os

import flax.linen as nn
import jax
import optax
import wandb
import yaml
from etils import epath
from jax import config as jax_config
from jax import numpy as jnp
from orbax import checkpoint as ocp

from tai.models.transformer import Transformer, TransformerConfig
from tai.utils import metrics
from tai.utils.data import TfDatasetIterator, create_dataset

# Configuring JAX & tensorflow
jax_config.update("jax_debug_nans", True)
jax_config.update("jax_debug_infs", True)


# Loading configuration
root_dir = epath.Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
config_dir = root_dir / "config" / "transformer.yaml"
with open(config_dir, "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)
    del config_dir


seed = config["seed"]
model_config = config["model"]
train_config = config["training"]
tok_config = config["tokenizer"]
wandb_config = config["wandb"]
checkpoint_config = config["checkpoint"]
optimizer_config = config["optimizer"]
del config


# Initialize dataset
tok_config["path"] = root_dir / tok_config["path"]
train_config["cache_dir"] = root_dir / train_config["cache_dir"]
if not train_config["cache_dir"].exists():
    train_config["cache_dir"].mkdir(parents=True, exist_ok=True)
src = root_dir / train_config["path"] / train_config["src"]
tgt = root_dir / train_config["path"] / train_config["tgt"]
dataloader = TfDatasetIterator(
    create_dataset(train_config, tok_config, src, tgt),
    checkpoint_dir=checkpoint_config["path"],
)
del src, tgt


# Initialize transformer model
param_key, dropout_key = jax.random.split(jax.random.PRNGKey(seed))
model = Transformer(TransformerConfig.fromDict(model_config))
params = model.init(
    {
        "params": param_key,
        "dropout": dropout_key,
    },
    jnp.ones((train_config["src_max_len"],), dtype=jnp.int32),
    jnp.ones((train_config["tgt_max_len"],), dtype=jnp.int32),
)


# Defining optimiser
lr = optax.warmup_cosine_decay_schedule(
    optimizer_config["lr"],
    optimizer_config["peak_lr"],
    optimizer_config["warmup_steps"],
    optimizer_config["decay_steps"],
    optimizer_config["final_lr"],
)
optimizer = optax.adam(lr)
del lr


# Initialize training state
state = metrics.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer,
    metrics=metrics.Metrics.empty(),
)
del params, optimizer


# Define a train step function
@jax.jit
def train_step(state, inputs, targets, labels):
    def loss_fn(params):
        preds = state.apply_fn(params, inputs, targets, rngs={"dropout": dropout_key})
        return jnp.mean(
            jax.vmap(metrics.cross_entropy_with_integer_labels)(preds, labels)
        )

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return state.apply_gradients(grads=grads), loss


@jax.jit
def compute_metrics(state, inputs, targets, labels):
    preds = state.apply_fn(state.params, inputs, targets, rngs={"dropout": dropout_key})
    loss = jnp.mean(jax.vmap(metrics.cross_entropy_with_integer_labels)(preds, labels))
    accuracy = jnp.mean(jax.vmap(metrics.accuracy)(preds, labels))
    metrics_updates = state.metrics.single_from_model_output(
        loss=loss,
        accuracy=accuracy,
    )
    metrics = state.metrics.merge(metrics_updates)
    return state.replace(metrics=metrics)


# Make checkpoint directory if it doesn't exist and initialize checkpoint manager
if use_checkpoint := checkpoint_config["use_checkpoint"]:
    freq = checkpoint_config["freq"]
    checkpoint_path = root_dir / checkpoint_config["path"]
    if checkpoint_path.exists() and checkpoint_config["overwrite"]:
        checkpoint_path.rmtree()
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=checkpoint_config["max_to_keep"],
        save_interval_steps=freq,
    )
    mngr = ocp.CheckpointManager(
        checkpoint_path.resolve(),
        options=options,
    )
    del checkpoint_path, options

    # Loading state and dataloader if checkpoint exists
    if mngr.latest_step() is not None:
        mngr.wait_until_finished()
        state = mngr.restore(mngr.latest_step())
        dataloader = dataloader.restore("dataset")


# Optional configuration for logging to wandb
if use_wandb := wandb_config["use_wandb"]:
    run_config = wandb_config["run_config"]
    run_config["epochs"] = train_config["epochs"]
    run_config["model_parameters"] = sum(
        [x.size for x in jax.tree_util.tree_leaves(params)]
    )
    run_config = dict(**run_config, **model_config)

    run = wandb.init(
        project=wandb_config["project"],
        notes=wandb_config["notes"],
        name=wandb_config["name"],
        config=run_config,
    )
    del wandb_config


# Training loop
print("Running...")
for i, (inputs, targets, labels) in enumerate(dataloader):
    # Train step
    state, loss = train_step(state, inputs, targets, labels)

    # Compute metrics
    state = compute_metrics(state, inputs, targets, labels)
    metrics = state.metrics.compute()

    # Checkpoint model and optimiser state
    if use_checkpoint and i % freq == 0:
        mngr.save(i, args=ocp.args.PyTreeSave(item=state))
        dataloader.save("dataset")

    # Log to wandb
    if use_wandb:
        wandb.log({"loss": loss, **metrics})
