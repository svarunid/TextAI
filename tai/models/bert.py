from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class BertConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    qkv_dim: int
    mlp_dim: int
    emb_dim: int
    max_length: int
    dropout_rate: float
    deterministic: bool = False
    kernel_init: nn.initializers = nn.initializers.he_uniform()
    bias_init: nn.initializers = nn.initializers.ones

    @staticmethod
    def fromDict(config: Dict):
        return BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            qkv_dim=config["qkv_dim"],
            mlp_dim=config["mlp_dim"],
            emb_dim=config["emb_dim"],
            max_length=config["max_length"],
            dropout_rate=config["dropout_rate"],
            deterministic=config["deterministic"],
        )


class BertEmbeddings(nn.Module):
    config: BertConfig

    def setup(self):
        self.token_emb = nn.Embed(
            self.config.vocab_size,
            self.config.emb_dim,
            kernel_init=self.config.kernel_init,
        )
        self.seq_emb = nn.Embed(
            2, self.config.emb_dim, kernel_init=self.config.kernel_init
        )
        self.pos_emb = nn.Embed(
            self.config.max_length,
            self.config.emb_dim,
            kernel_init=self.config.kernel_init,
        )

    def __call__(self, inputs, sequence_id) -> jax.Array:
        token_emb = self.token_emb(inputs)
        seq_emb = self.seq_emb(sequence_id)
        pos_emb = self.pos_emb(jnp.arange(self.config.max_length))
        return token_emb + seq_emb + pos_emb


class MLP(nn.Module):
    config: BertConfig

    @nn.compact
    def __call__(self, inputs) -> jax.Array:
        x = nn.Dense(
            features=self.config.mlp_dim,
            use_bias=False,
            kernel_init=self.config.kernel_init,
        )(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )
        x = nn.Dense(
            features=self.config.emb_dim,
            use_bias=False,
            kernel_init=self.config.kernel_init,
        )(x)
        x = nn.Dropout(rate=self.config.dropout)(
            x, deterministic=self.config.deterministic
        )
        return x


class EncoderLayer(nn.Module):
    config: BertConfig

    @nn.compact
    def __call__(self, inputs, mask) -> jax.Array:
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.qkv_dim,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
            broadcast_dropout=False,
            dropout_rate=self.config.dropout,
            deterministic=self.config.deterministic,
        )(inputs, inputs, inputs, mask=mask)
        x = nn.LayerNorm()(x + inputs)

        y = MLP(self.config)(x)
        y = nn.LayerNorm()(y + x)
        return y


class Encoder(nn.Module):
    config: BertConfig

    @nn.compact
    def __call__(self, inputs, mask, sequence_id) -> jax.Array:
        inputs, sequence_id = [x.astype(np.int32) for x in (inputs, sequence_id)]
        x = BertEmbeddings(self.config)(inputs, sequence_id)
        for i in range(self.config.num_layers):
            x = EncoderLayer(self.config, name=f"encoder_layer_{i}")(x, mask)
        return x


class Bert(nn.Module):
    config: BertConfig

    @nn.compact
    def __call__(self, inputs, sequence_id) -> jax.Array:
        mask = nn.make_attention_mask(inputs > 0, inputs > 0)
        x = Encoder(self.config)(inputs, mask, sequence_id)
        x = nn.Dense(
            features=self.config.emb_dim,
            kernel_init=self.config.kernel_init,
            bias_init=self.config.bias_init,
        )(x)
        return x


def create_model(config: BertConfig, rngs: Dict) -> Bert:
    """
    Creates and initializes a Bert model with the given configuration and random number generators.

    Args:
        config (BertConfig): Configuration for the Bert model
        rngs (Dict): Random number generators for initializing the model

    Returns:
        tuple: A tuple containing the initialized Bert model and its parameters
    """
    model = Bert(config)
    inputs = jnp.ones((config.max_length,), dtype=np.int32)
    sequence_id = jnp.ones((config.max_length,), dtype=np.int32)
    params = model.init(rngs, inputs, sequence_id)
    return model, params
