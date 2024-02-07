import flax.linen as nn
from clu import metrics
from flax import struct
from flax.training.train_state import TrainState as _TrainState # prevent name conflict
from jax import numpy as jnp


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    accuracy: metrics.Average.from_output("accuracy")


class TrainState(_TrainState):
    metrics: Metrics

    @classmethod
    def create(cls, apply_fn, params, tx):
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            metrics=Metrics.empty(),
            opt_state=opt_state,
        )

def cross_entropy_with_label_smoothing(preds, labels, smoothing=0.1):
    """
    Cross-entropy loss with label smoothing for padded interger labels.

    Args:
        preds (jnp.ndarray): Predictions.
        labels (jnp.ndarray): Labels.
        smoothing (float): Smoothing factor.

    Returns:
        jnp.ndarray: Cross-entropy loss.
    """
    mask = labels != 0
    confidence = 1.0 - smoothing
    n_class = preds.shape[-1]
    smooth_labels = jnp.where(
        nn.one_hot(labels, n_class) == 0, smoothing / (n_class - 1), confidence
    )
    smooth_labels.at[..., 0].set(0.0)
    prod = jnp.sum(smooth_labels * nn.log_softmax(preds), axis=-1) * mask
    return -jnp.sum(prod) / jnp.sum(mask)


def cross_entropy_with_integer_labels(preds, labels):
    """
    Cross-entropy loss for padded integer labels.

    Args:
        preds (jnp.ndarray): Predictions.
        labels (jnp.ndarray): Labels.

    Returns:
        jnp.ndarray: Cross-entropy loss.
    """
    mask = labels != 0
    ce = jnp.take_along_axis(nn.log_softmax(preds), labels[..., None], axis=-1)
    return -jnp.sum(ce * mask) / jnp.sum(mask)

def accuracy(preds, labels):
    """
    Accuracy for padded integer labels.

    Args:
        preds (jnp.ndarray): Predictions.
        labels (jnp.ndarray): Labels.

    Returns:
        jnp.ndarray: Accuracy.
    """
    mask = labels != 0
    correct = jnp.allclose(jnp.argmax(preds, axis=-1), labels) * mask
    return jnp.sum(correct) / jnp.sum(mask)

