import jax
import numpy as np
import equinox as eqx
import jax.numpy as jnp

from .linear import Linear, FFNN
from .norm import LayerNorm

class MultiHeadAttention(eqx.Module):
    """
    Multi-Head Attention Layer
    """
    wquery: jax.Array
    wkey: jax.Array
    wvalue: jax.Array
    weights: jax.Array
    n_heads: int = eqx.field(static=True)
    dim_k: int = eqx.field(static=True)

    def __init__(self, key, n_heads, dim):
        """
        Initializes the weights for the layer.

        Parameters
        ----------
        key: jax.random.PRNGKey
            Random key for initializing the weights.
        n_heads: int
            Number of heads.
        dim: int
            Model dimension.
        """
        if (dim % n_heads) != 0:
            raise ValueError("Model dimensions must be a multiple of no. of heads")
        dim_k = dim // n_heads
        init = jax.nn.initializers.he_uniform()
        wkey, qkey, kkey, vkey = jax.random.split(key, num=4)
        self.weights = init(key=wkey, shape=(n_heads * dim_k, dim))
        self.wquery = init(key=qkey, shape=(dim, dim))
        self.wkey = init(key=kkey,shape=(dim, dim))
        self.wvalue = init(key=vkey, shape=(dim, dim))
        self.n_heads = n_heads
        self.dim_k = dim_k

    @eqx.filter_jit
    def __call__(self, query, key, value, mask):
        """
        Performs the attention operation.

        Parameters
        ----------
        query: jax.Array
            Query array.
        key: jax.Array
            Key array.
        value: jax.Array
            Value array.
        mask: jax.Array
            Mask array.
        
        Returns
        -------
        jax.Array
            Attention array.
        """
        query, key, value = query @ self.wquery, key @ self.wkey, value @ self.wvalue
        query, key, value = [jnp.transpose(jnp.reshape(x, (-1, self.n_heads, self.dim_k)), (1, 0, 2)) 
                             for x in (query, key, value)]
        if mask.ndim == 1:
            mask = jnp.expand_dims(mask, axis=0)
        # Attention Calculation
        scaled_dot_prod = query @ jnp.transpose(key, (0, 2, 1)) / jnp.sqrt(query.shape[-1])
        scaled_dot_prod = mask + scaled_dot_prod
        attn = jax.nn.softmax(scaled_dot_prod) @ value
        return jnp.reshape(jnp.transpose(attn, (1, 0, 2)), (-1, self.n_heads * self.dim_k)) @ self.weights

class Encoder(eqx.Module):
    """
    Encoder Layer.
    """
    emb: jax.Array
    attn_layers: list
    ff_layers:list
    attn_norms: list
    ff_norms: list
    n_layers: int = eqx.field(static=True)
    pe: jax.Array = eqx.field(static=True)

    def __init__(self, key, n_layers, n_heads, dim, seq_len, vocab):
        """
        Initializes the weights for the layer.

        Parameters
        ----------
        key: jax.random.PRNGKey
            Random key for initializing the weights.
        n_layers: int
            Number of layers.
        n_heads: int
            Number of heads.
        dim: int
            Model dimension.
        seq_len: int
            Maximum sequence length.
        vocab: int
            Vocabulary size.
        """
        keys = jax.random.split(key, num=n_layers*2+1)
        emb_key, attn_keys, ff_keys = keys[0], keys[1:n_layers+1], keys[n_layers+1:]
        self.emb = jax.random.normal(emb_key, (vocab, dim))
        # Self-Attention & Forward Layers
        self.attn_layers = [MultiHeadAttention(key, n_heads, dim) for key in attn_keys]
        self.ff_layers = [FFNN(key, dim, dim, dim*2) for key in ff_keys]
        # Layer Norms
        self.attn_norms = [LayerNorm(dim) for _ in range(n_layers)]
        self.ff_norms = [LayerNorm(dim) for _ in range(n_layers)]
        # Positional Encodings
        pos = jnp.arange(seq_len)[:, jnp.newaxis]
        div_term = 10_000 ** (2 * jnp.arange(0, dim, 2) / dim)
        pe = jnp.empty((seq_len, dim))
        pe = pe.at[:, 0::2].set(jnp.sin(pos / div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(pos / div_term))
        self.pe = pe
        # Static Arguments
        self.n_layers = n_layers

    @eqx.filter_jit
    def __call__(self, x, mask):
        """
        Performs the attention operation.

        Parameters
        ----------
        x: jax.Array
            Input array.
        mask: jax.Array
            Mask array.
        
        Returns
        -------
        jax.Array
            Attention array.
        """
        x = self.emb[x[...]]
        x = x + self.pe
        for i in range(self.n_layers):
            x = self.attn_norms[i](self.attn_layers[i](x, x, x, mask) + x)
            x = self.ff_norms[i](self.ff_layers[i](x) + x)
        return x

class Decoder(eqx.Module):
    """
    Decoder Layer.
    """
    emb: jax.Array
    mask: jax.Array = eqx.field(static=True)
    masked_attn_layers: list
    attn_layers: list
    ff_layers:list
    masked_attn_norms: list
    attn_norms: list
    ff_norms: list
    n_layers: int = eqx.field(static=True)
    pe: jax.Array = eqx.field(static=True)

    def __init__(self, key, n_layers, n_heads, dim, seq_len, vocab):
        """
        Initializes the weights for the layer.

        Parameters
        ----------
        key: jax.random.PRNGKey
            Random key for initializing the weights.
        n_layers: int
            Number of layers.
        n_heads: int
            Number of heads.
        dim: int
            Model dimension.
        seq_len: int
            Maximum sequence length.
        vocab: int
            Vocabulary size.
        """
        keys = jax.random.split(key, num=n_layers*3+1)
        emb_key, attn_keys, ff_keys, masked_attn_keys = (
            keys[0], keys[1:n_layers+1], keys[n_layers+1:n_layers*2+1], keys[n_layers*2+1:]
        )
        self.emb = jax.random.normal(emb_key, (vocab, dim))
        self.mask = jnp.where(jnp.triu(jnp.ones((seq_len, seq_len)), 1) == 1, np.NINF, 0)
        # Masked-Attention, Self-Attention & Forward Layers
        self.masked_attn_layers = [MultiHeadAttention(key, n_heads, dim) for key in masked_attn_keys]
        self.attn_layers = [MultiHeadAttention(key, n_heads, dim) for key in attn_keys]
        self.ff_layers = [FFNN(key, dim, dim, dim*2) for key in ff_keys]
        # Layer Norms
        self.masked_attn_norms = [LayerNorm(dim) for _ in range(n_layers)]
        self.attn_norms = [LayerNorm(dim) for _ in range(n_layers)]
        self.ff_norms = [LayerNorm(dim) for _ in range(n_layers)]
        # Positional Encodings
        pos = jnp.arange(seq_len)[:, jnp.newaxis]
        div_term = 10_000 ** (2 * jnp.arange(0, dim, 2) / dim)
        pe = jnp.empty((seq_len, dim))
        pe = pe.at[:, 0::2].set(jnp.sin(pos / div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(pos / div_term))
        self.pe = pe
        # Static Arguments
        self.n_layers = n_layers

    @eqx.filter_jit
    def __call__(self, x, m, x_mask, m_mask):
        """
        Performs the attention operation.

        Parameters
        ----------
        x: jax.Array
            Input array.
        m: jax.Array
            Memory array.
        x_mask: jax.Array
            Input mask array.
        m_mask: jax.Array
            Memory mask array.

        Returns
        -------
        jax.Array
            Attention array.
        """
        x = self.emb[x[...]]
        x = x + self.pe
        x_mask = self.mask + x_mask
        for i in range(self.n_layers):
            x = self.masked_attn_norms[i](self.masked_attn_layers[i](x, x, x, x_mask) + x)
            x = self.attn_norms[i](self.attn_layers[i](x, m, m, m_mask) + x)
            x = self.ff_norms[i](self.ff_layers[i](x) + x)
        return x

class EncoderDecoder(eqx.Module):
    """
    Encoder-Decoder Model.
    """
    encoder: eqx.Module
    decoder: eqx.Module

    def __init__(self, key, dim, enc_layers, dec_layers, n_heads, in_vocab, out_vocab, in_len, out_len):
        """
        Initializes the weights for the layer.

        Parameters
        ----------
        key: jax.random.PRNGKey
            Random key for initializing the weights.
        dim: int
            Model dimension.
        enc_layers: int
            Number of encoder layers.
        dec_layers: int
            Number of decoder layers.
        n_heads: int
            Number of heads.
        in_vocab: int
            Input vocabulary size.
        out_vocab: int
            Output vocabulary size.
        in_len: int
            Maximum input sequence length.
        out_len: int
            Maximum output sequence length.
        """
        enc_key, dec_key = jax.random.split(key, num=2)
        self.encoder = Encoder(enc_key, enc_layers, n_heads, dim, in_len, in_vocab)
        self.decoder = Decoder(dec_key, dec_layers, n_heads, dim, out_len, out_vocab)

    @eqx.filter_jit
    def __call__(self, X, y, X_mask, y_mask):
        """
        Performs the attention operation.

        Parameters
        ----------
        X: jax.Array
            Input array.
        y: jax.Array
            Target array.
        X_mask: jax.Array
            Input mask array.
        y_mask: jax.Array
            Target mask array.
        
        Returns
        -------
        jax.Array
            Attention array.
        """
        m = self.encoder(X, X_mask)
        h = self.decoder(y, m, y_mask, X_mask)
        return h

class Transformer(eqx.Module):
    """
    Transformer Model.
    """
    enc_dec: eqx.Module
    linear: eqx.Module

    def __init__(self, key, dim, enc_layers, dec_layers, n_heads, in_vocab, out_vocab, in_len, out_len):
        """
        Initializes the weights for the layer.

        Parameters
        ----------
        key: jax.random.PRNGKey
            Random key for initializing the weights.
        dim: int
            Model dimension.
        enc_layers: int
            Number of encoder layers.
        dec_layers: int
            Number of decoder layers.
        n_heads: int 
            Number of heads.
        in_vocab: int
            Input vocabulary size.
        out_vocab: int
            Output vocabulary size.
        in_len: int
            Maximum input sequence length.
        out_len: int
            Maximum output sequence length.
        """
        encdec_key, linear_key = jax.random.split(key)
        self.enc_dec = EncoderDecoder(encdec_key, dim,
                                      enc_layers, dec_layers, n_heads,
                                      in_vocab, out_vocab, in_len, out_len)
        self.linear = Linear(linear_key, dim, out_vocab)

    @eqx.filter_jit
    def __call__(self, X, y, X_mask, y_mask):
        """
        Performs the attention operation. Outputs the prediction probabilities.

        Parameters
        ----------
        X: jax.Array
            Input array.
        y: jax.Array
            Target array.
        X_mask: jax.Array
            Input mask array.
        y_mask: jax.Array
            Target mask array.
        """
        logits = self.enc_dec(X, y, X_mask, y_mask)
        return jax.nn.softmax(self.linear(logits)) + 1e-9