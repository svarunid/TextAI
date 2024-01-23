import optax
import tensorflow as tf
from clu import metrics
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from jax import numpy as jnp


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    accuracy: metrics.Accuracy


class TrainState(TrainState):
    metrics: Metrics


# def loss_fn(rngs, state, inputs, targets, labels):
#     preds = state.apply_fn({"params": state.params}, inputs, targets, rngs=rngs)
#     preds = nn.log_softmax(preds)
#     mask = jnp.not_equal(jnp.sum(labels, axis=-1), 0)
#     preds = preds[mask]
#     labels = labels[mask]
#     return -jnp.sum(preds * labels, axis=-1)


def smooth_labels(labels, num_labels, smoothing=0.1):
    labels = tf.expand_dims(labels, axis=-1)
    labels = tf.one_hot(labels, depth=num_labels)
    return tf.where(
        labels[..., 0] == 1,
        tf.zeros(num_labels),
        (1 - smoothing) * labels / (num_labels - 1),
    )
