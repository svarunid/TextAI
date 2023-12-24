import jax
import jax.numpy as jnp
import equinox as eqx

class LayerNorm(eqx.Module):
    """
    Layer normalization.
    """
    gamma: jax.Array
    bias: jax.Array
    eps: int = eqx.field(static=True)

    def __init__(self, size, eps=1e-6):
        """
        Initialize layer normalization.

        Parameters
        ----------
        size: int
            Size of the input.
        eps: float, optional (default is 1e-6)
            Epsilon value for numerical stability.
        """
        self.gamma = jnp.ones(size)
        self.bias = jnp.ones(size)
        self.eps = 1e-6

    @eqx.filter_jit
    def __call__(self, x):
        """
        Apply layer normalization to the input.

        Parameters
        ----------
        x: jax.Array
            Input array.
        
        Returns
        -------
        jax.Array
            Normalized output array.
        """
        mean = jnp.mean(x, -1, keepdims=True)
        std = jnp.std(x, -1, keepdims=True)
        return (self.gamma * (x - mean) / (std + self.eps)) + self.bias