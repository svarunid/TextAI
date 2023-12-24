import jax
import equinox as eqx
import jax.numpy as jnp

class Linear(eqx.Module):
    """
    Linear layer with optional bias.
    """
    weights: jax.Array
    bias: jax.Array
    use_bias: bool = eqx.field(static=True)

    def __init__(self, key, nin, nout, use_bias=False):
        """
        Initialize a linear layer with optional bias.

        Parameters
        ----------
        key: jax.random.PRNGKey
            Random key for initializing weights.
        nin: int
            Number of input features.
        nout: int
            Number of output features.
        use_bias: bool, optional (default is False)
            Whether to use bias or not.
        """
        init = jax.nn.initializers.he_uniform()
        self.weights = init(key=key, shape=(nin, nout))
        if use_bias:
            self.bias = jnp.ones(nout)
        else: 
            self.bias = None
        self.use_bias = use_bias
        
    @eqx.filter_jit
    def __call__(self, x):
        """
        Apply linear transformation to the input.

        Parameters
        ----------
        x: jax.Array
            Input array.
        
        Returns
        -------
        jax.Array
            Output array.
        """
        x = x @ self.weights
        if self.use_bias:
            x = x + self.bias
        return x

class FFNN(eqx.Module):
    """
    Feed forward neural network with optional bias.
    """
    layers: list
    def __init__(self, key, nin, nout, nhidden, n_layers=2, use_bias=False):
        """
        Initialize a feed forward neural network with optional bias.

        Parameters
        ----------
        key: jax.random.PRNGKey
            Random key for initializing weights.
        nin: int
            Number of input features.
        nout: int
            Number of output features.
        nhidden: int
            Number of hidden features.
        n_layers: int, optional (default is 2)
            Number of layers in the network.
        use_bias: bool, optional (default is False)
            Whether to use bias or not.
        """
        keys = jax.random.split(key, num=n_layers)
        layers = [
            Linear(keys[0], nin, nhidden, use_bias)
        ]
        for i in range(1, n_layers-1):
            layers.append(jax.nn.gelu)
            layers.append(Linear(keys[i], nhidden, nhidden, use_bias))
        if n_layers != 1:
            layers.append(Linear(keys[-1], nhidden, nout, use_bias))
        self.layers = layers

    @eqx.filter_jit
    def __call__(self, x):
        """
        Apply feed forward neural network to the input.

        Parameters
        ----------
        x: jax.Array
            Input array.

        Returns
        -------
        jax.Array
            Output array.
        """
        for layer in self.layers:
            x = layer(x)
        return x