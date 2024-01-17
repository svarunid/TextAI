import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from typing import Callable, Any, Optional


def sinusoidal_init(max_len: int, max_scale: float = 1., min_scale: float = 10_000.) -> jax.Array:
    def init(key, shape, dtype=np.float32):
        del key, dtype
        dim = shape[-1]
        position = np.arange(0, max_len)[:, jnp.newaxis]
        scale_factor = (max_scale / min_scale)  ** (2 * jnp.arange(0, dim, 2) / dim)
        div_term = np.divide(1, scale_factor)
        pe = np.empty((max_len, dim), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]
        return jnp.array(pe)
    return init

@struct.dataclass
class TransformerConfig:
    in_vocab: int
    out_vocab: int
    emb_dim: int
    num_heads: int
    num_layers: int
    qkv_dim: int
    mlp_dim: int
    max_len: int
    dropout: float
    deterministic: bool = False
    kernel_init: nn.initializers = nn.initializers.he_uniform()
    bias_init: nn.initializers = nn.initializers.ones
    pos_emb_init: Optional[Callable] = None

class PosEmbedding(nn.Module):
    config: TransformerConfig
    
    @nn.compact
    def __call__(self, inputs) -> jax.Array:
        length = inputs.shape[1]
        pos_emb_shape = (1, length, self.config.emb_dim)
        if self.config.pos_emb_init is None:
            pos_emb = sinusoidal_init(self.config.max_len)(None, pos_emb_shape)
        else:
            pos_emb = self.param(
                'pos_emb', self.config.pos_emb_init, pos_emb_shape
            )
        pe = pos_emb[:, :length, :]
        return inputs + pe
        