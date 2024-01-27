import flax.linen as nn
from clu import metrics
from flax import struct
from flax.training.train_state import TrainState
from jax import lax
from jax import numpy as jnp


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    accuracy: metrics.Average.from_output("accuracy")


class TrainState(TrainState):
    metrics: Metrics


def cross_entropy_with_label_smoothing(preds, labels, smoothing=0.1):
    confidence = 1.0 - smoothing
    count = jnp.count_nonzero(labels)
    preds = nn.log_softmax(preds)
    n_class = preds.shape[-1]
    smooth_labels = jnp.where(
        nn.one_hot(labels, n_class) == 0, smoothing / (n_class - 1), confidence
    )
    smooth_labels.at[..., 0].set(0.0)
    prod = lax.dynamic_slice(smooth_labels * preds, (0, 0), (count, n_class))
    return -jnp.sum(prod) / count


def cross_entropy_with_integer_labels(preds, labels):
    preds = nn.log_softmax(preds)
    preds = jnp.where(labels == 0, 0, jnp.take(preds, labels, axis=-1))
    count = jnp.count_nonzero(labels)
    return -jnp.sum(preds) / count


def accuracy(preds, labels):
    preds = nn.softmax(preds)
    preds = jnp.where(labels == 0, 0, jnp.argmax(preds, axis=-1))
    count = jnp.count_nonzero(labels)
    return jnp.sum(preds == labels) / count
