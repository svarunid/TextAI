import os

import jax
import wandb
import yaml
from etils import epath
from jax import config as jax_config
from jax import numpy as jnp
from orbax import checkpoint as ocp

from tai.models.transformer import TransformerConfig, create_model
from tai.utils import checkpoint, data, optim, wandb
from tai.utils.metrics import (
    TrainState,
    accuracy as accuracy_fn,
    cross_entropy_with_integer_labels,
)

# Configuring JAX & tensorflow
jax_config.update("jax_debug_nans", True)
jax_config.update("jax_debug_infs", True)


# Loading configuration
root_dir = epath.Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
config_dir = root_dir / "config" / "transformer.yml"
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
dataset = data.create_dataset(root_dir, train_config, tok_config)

# Initialize transformer model and optimizer
params_key, dropout_key = jax.random.split(jax.random.PRNGKey(seed))
model, params = create_model(
    TransformerConfig.fromDict(model_config),
    {
        "params": params_key, 
        "dropout": dropout_key
    },
)
optimizer = optim.optimizer(optimizer_config)


# Initialize training state
state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer,
)
del params, optimizer


# Define a train step function
@jax.jit
def train_step(state, inputs, targets, labels):
    def loss_fn(params):
        preds = state.apply_fn(params, inputs, targets, rngs={"dropout": dropout_key})
        return jnp.mean(jax.vmap(cross_entropy_with_integer_labels)(preds, labels))

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


@jax.jit
def compute_metrics(state, inputs, targets, labels):
    preds = state.apply_fn(state.params, inputs, targets, rngs={"dropout": dropout_key})
    loss = jnp.mean(jax.vmap(cross_entropy_with_integer_labels)(preds, labels))
    accuracy = jnp.mean(jax.vmap(accuracy_fn)(preds, labels))
    metrics_updates = state.metrics.single_from_model_output(
        loss=loss,
        accuracy=accuracy,
    )
    metrics = state.metrics.merge(metrics_updates)
    return state.replace(metrics=metrics)


# Make checkpoint directory if it doesn't exist and initialize checkpoint manager
periodic_checkpoint_loader = checkpoint.PeriodicCheckpoint(
    root_dir, checkpoint_config, dataset
)

if use_checkpoint := checkpoint_config["use_checkpoint"]:
    # Loading state and dataloader if checkpoint exists
    if periodic_checkpoint_loader.latest_step() is not None:
        state, dataloader = periodic_checkpoint_loader.restore(
            periodic_checkpoint_loader.latest_step()
        )


# Optional configuration for logging to wandb
if use_wandb := wandb_config["use_wandb"]:
    periodic_log = wandb.configure_wandb(wandb_config, model_config)


# Training loop
print("Running...")
for i, (inputs, targets, labels) in enumerate(periodic_checkpoint_loader):
    # Train step
    state = train_step(state, inputs, targets, labels)

    # Compute metrics
    state = compute_metrics(state, inputs, targets, labels)
    metrics = state.metrics.compute()

    print(f"Loss: {metrics['loss']}, Accuracy: {metrics['accuracy']}")

    # Checkpoint model and optimiser state
    if use_checkpoint:
        periodic_checkpoint_loader.save(i, args=ocp.args.PyTreeSave(item=state))

    # Log to wandb
    if use_wandb:
        periodic_log(
            step=i,
            data={
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
            },
        )
