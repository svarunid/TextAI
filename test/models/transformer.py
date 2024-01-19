from dataclasses import FrozenInstanceError

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from tai.models.transformer import PosEmbedding, Transformer, TransformerConfig


@pytest.fixture
def config():
    return TransformerConfig(
        in_vocab=100,
        out_vocab=100,
        emb_dim=128,
        qkv_dim=128,
        num_heads=8,
        mlp_dim=1024,
        num_layers=2,
        dropout=0.1,
        max_len=30,
        pos_emb_init=nn.initializers.he_uniform(),
    )


@pytest.fixture
def transformer(config):
    return Transformer(config)


@pytest.fixture
def pos_embedding(config):
    return PosEmbedding(config)


@pytest.fixture
def key():
    return jax.random.key(0)


@pytest.fixture
def inputs():
    return jnp.ones((1, 30, 128))


def test_transformer_config_creation(config):
    assert config.in_vocab == 100
    assert config.out_vocab == 100
    assert config.emb_dim == 128
    assert config.num_heads == 8
    assert config.num_layers == 2
    assert config.qkv_dim == 128
    assert config.mlp_dim == 1024
    assert config.max_len == 30
    assert config.dropout == 0.1
    assert config.deterministic == False
    assert config.decode == False

    with pytest.raises(FrozenInstanceError):
        config.in_vocab = 200


def test_transformer_config_from_dict():
    config_dict = {
        "in_vocab": 100,
        "out_vocab": 100,
        "emb_dim": 128,
        "num_heads": 8,
        "num_layers": 2,
        "qkv_dim": 128,
        "mlp_dim": 1024,
        "max_len": 30,
        "dropout": 0.1,
        "deterministic": True,
        "decode": True,
    }

    config = TransformerConfig.fromDict(config_dict)

    assert config.in_vocab == 100
    assert config.out_vocab == 100
    assert config.emb_dim == 128
    assert config.num_heads == 8
    assert config.num_layers == 2
    assert config.qkv_dim == 128
    assert config.mlp_dim == 1024
    assert config.max_len == 30
    assert config.dropout == 0.1
    assert config.deterministic == True
    assert config.decode == True

    with pytest.raises(FrozenInstanceError):
        config.in_vocab = 200


class TestPosEmbedding:
    def test_pos_embedding_shape_inference(self, key, inputs, pos_embedding):
        output, _ = pos_embedding.init_with_output(key, inputs)
        assert output.shape == (1, 30, 128)

    def test_pos_embedding_dynamic(self, key, inputs, pos_embedding):
        output, variables = pos_embedding.init_with_output(key, inputs)
        assert output.shape == (1, 30, 128)
        assert "params" in variables.keys()

    def test_pos_embedding_static(self, key, inputs):
        config = TransformerConfig(
            in_vocab=100,
            out_vocab=100,
            emb_dim=128,
            qkv_dim=128,
            num_heads=8,
            mlp_dim=1024,
            num_layers=2,
            dropout=0.1,
            max_len=30,
        )
        pos_embedding = PosEmbedding(config)
        output, variables = pos_embedding.init_with_output(key, inputs)
        assert output.shape == (1, 30, 128)
        assert list(variables.keys()) == []

    def test_pos_embedding_cache(self, key, inputs):
        config = TransformerConfig(
            in_vocab=100,
            out_vocab=100,
            emb_dim=128,
            qkv_dim=128,
            num_heads=8,
            mlp_dim=1024,
            num_layers=2,
            dropout=0.1,
            max_len=30,
            decode=True,
        )
        pos_embedding = PosEmbedding(config)
        output, variables = pos_embedding.init_with_output(key, inputs)
        assert output.shape == (1, 30, 128)
        assert "cache" in variables.keys()
