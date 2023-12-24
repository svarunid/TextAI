import jax
import optax
import equinox as eqx
import jax.numpy as jnp
from jax.tree_util import tree_map

def optim(model, optimizer, loss_fn, vectorize=True, in_axes=None, out_axes=None):
    """
    Optimizer function for the model.

    Parameters
    ----------
    model : Transformer
        The model to optimize.
    optimizer : optax.GradientTransformation
        The optimizer to use.
    loss_fn : function
        The loss function.
    vectorize : bool, optional
        Whether to vectorize the loss function or not, by default True.
    in_axes : tuple, optional
        The input axes for the loss function, by default None.
    out_axes : int, optional
        The output axes for the loss function, by default None.
    
    Returns
    -------
    opt_state : optax.OptState
        The optimizer state.
    step : Callable
        The optimizer step function.
    """
    opt_state = optimizer.init(model)
    grad = jax.value_and_grad(loss_fn)
    if vectorize:
        gradient = jax.vmap(grad, in_axes=in_axes, out_axes=out_axes)
    
    @eqx.filter_jit
    def step(model, opt_state, X, y, X_mask, y_mask, labels):
        """
        Optimizer step function.

        Parameters
        ----------
        model : Transformer
            The model to optimize.
        opt_state : optax.OptState
            The optimizer state.
        X : jnp.ndarray
            The input sequence.
        y : jnp.ndarray
            The output sequence.
        X_mask : jnp.ndarray
            The input sequence mask.
        y_mask : jnp.ndarray
            The output sequence mask.
        labels : jnp.ndarray
            The output sequence labels.

        Returns
        -------
        model : Transformer
            The optimized model.
        opt_state : optax.OptState
            The optimizer state.
        loss_value : float
            The loss value.
        """
        loss_value, grads = gradient(model, X, y, X_mask, y_mask, labels)
        if vectorize:
            loss_value = jnp.mean(loss_value)
            grads = tree_map(lambda x: jnp.mean(x, axis=0), grads)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)
        return model, opt_state, loss_value

    return opt_state, step