import jax
from jax import numpy as jnp
from spax.nn.linear import Embedding, Linear, embedding, linear, Sequential, sequential
from spax.nn.transformers import decoder_layer, encoder_layer
from spax.struct import field, struct


@struct
class Transformer:
    in_emb: Embedding
    out_emb: Embedding
    encoder: Sequential
    decoder: Sequential
    linear: Linear
    in_pe: jax.Array = field(leaf=False)
    out_pe: jax.Array = field(leaf=False)


def transformer(
    key,
    in_vocab_size,
    out_vocab_size,
    in_seq_len,
    out_seq_len,
    d_model,
    n_heads,
    d_ff,
    enc_layers,
    dec_layers,
):
    in_emb_key, out_emb_key, enc_key, dec_key, linear_key = jax.random.split(key, num=5)
    in_emb, emb_forward = embedding(in_emb_key, in_vocab_size, d_model)
    out_emb, _ = embedding(out_emb_key, out_vocab_size, d_model)
    encoder_layers, enc_forward = sequential(
        *tuple(
            encoder_layer(enc_key, n_heads, d_model, d_ff) for _ in range(enc_layers)
        )
    )
    decoder_layers, dec_forward = sequential(
        *tuple(
            decoder_layer(dec_key, n_heads, d_model, d_ff) for _ in range(dec_layers)
        )
    )
    linear_layer, lin_forward = linear(linear_key, d_model, out_vocab_size)

    in_pos = jnp.arange(in_seq_len)[:, jnp.newaxis]
    out_pos = jnp.arange(out_seq_len)[:, jnp.newaxis]
    div_term = 10_000 ** (2 * jnp.arange(0, d_model, 2) / d_model)

    in_pe = jnp.empty((in_seq_len, d_model))
    in_pe = in_pe.at[:, 0::2].set(jnp.sin(in_pos / div_term))
    in_pe = in_pe.at[:, 1::2].set(jnp.cos(in_pos / div_term))

    out_pe = jnp.empty((out_seq_len, d_model))
    out_pe = out_pe.at[:, 0::2].set(jnp.sin(out_pos / div_term))
    out_pe = out_pe.at[:, 1::2].set(jnp.cos(out_pos / div_term))

    model = Transformer(
        in_emb, out_emb, encoder_layers, decoder_layers, linear_layer, in_pe, out_pe
    )

    @jax.jit
    def forward(model, X, y, X_mask, y_mask):
        X = emb_forward(model.in_emb, X)
        X = X + model.in_pe

        y = emb_forward(model.out_emb, y)
        y = y + model.out_pe

        X = enc_forward(model.encoder, X, common_args=(X_mask,))
        y = dec_forward(model.decoder, y, common_args=(X, y_mask, X_mask))

        return lin_forward(model.linear, y)

    return model, forward
