import flax.linen as nn
import optax
from clu import metrics
from flax import struct
from flax.training.train_state import TrainState
from jax import numpy as jnp


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    accuracy: metrics.Average.from_output("accuracy")


class TrainState(TrainState):
    metrics: Metrics


def cross_entropy_with_label_smoothing(preds, labels, smoothing=0.1):
    """
    NOTE: This function is not jit-able.
    """
    confidence = 1.0 - smoothing
    non_zero_count = jnp.count_nonzero(labels)
    preds, labels = preds[:non_zero_count], labels[:non_zero_count]
    true_dist = jnp.full_like(preds, smoothing / (preds.shape[-1] - 2))
    true_dist = true_dist.at[:, labels].set(confidence)
    true_dist.at[:, :2].set(0.0)
    return optax.softmax_cross_entropy(preds, true_dist).mean()


def cross_entropy_with_integer_labels(preds, labels):
    preds = nn.log_softmax(preds)
    preds = jnp.where(labels == 0, 0, jnp.take(preds, labels, axis=-1))
    count = jnp.count_nonzero(preds)
    return -jnp.sum(preds) / count


def accuracy(preds, labels):
    preds = nn.softmax(preds)
    preds = jnp.where(labels == 0, 0, jnp.argmax(preds, axis=-1))
    count = jnp.count_nonzero(labels)
    return jnp.sum(preds == labels) / count
