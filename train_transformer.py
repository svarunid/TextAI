import configparser
from os import path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import wandb
from jax.tree_util import tree_map
from spax.nn.linear import Embedding, Linear, embedding, linear
from spax.nn.transformers import Decoder, Encoder, decoder, encoder
from spax.nn.utils import optim
from spax.struct import field, struct

from helpers import spickle, text

config = configparser.ConfigParser()
config.read("./config/transformer_config.ini")

dataloader = spickle.sload(config.get("training", "data_path"))
dataloader = text.seq2seq_batched_iterator(
    dataloader,
    config.getint("model", "in_seq_len"),
    config.getint("model", "out_seq_len"),
    config.getint("model", "batch_size"),
)

key = jax.random.PRNGKey(config.getint("model", "seed"))

@struct
class Transformer:
    in_emb: Embedding
    out_emb: Embedding
    encoder: Encoder
    decoder: Decoder
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
    encoder_layer, enc_forward = encoder(enc_key, enc_layers, n_heads, d_model, d_ff)
    decoder_layer, dec_forward = decoder(dec_key, dec_layers, n_heads, d_model, d_ff)
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

    model = Transformer(in_emb, out_emb, encoder_layer, decoder_layer, linear_layer, in_pe, out_pe)

    @jax.jit
    def forward(model, X, y, X_mask, y_mask):
        X = emb_forward(model.in_emb, X)
        X = X + model.in_pe

        y = emb_forward(model.out_emb, y)
        y = y + model.out_pe

        X = enc_forward(model.encoder, X, X_mask)
        y = dec_forward(model.decoder, y, X, y_mask, X_mask)

        y = lin_forward(model.linear, y)
        return jax.nn.softmax(y)

    return model, forward


model, forward = transformer(
    key,
    config.getint("model", "in_vocab"),
    config.getint("model", "out_vocab"),
    config.getint("model", "in_seq_len"),
    config.getint("model", "out_seq_len"),
    config.getint("model", "d_model"),
    config.getint("model", "num_heads"),
    config.getint("model", "d_ff"),
    config.getint("model", "enc_layers"),
    config.getint("model", "dec_layers"),
)


def loss(model, X, y, X_mask, y_mask, labels):
    y_pred = jnp.log(forward(model, X, y, X_mask, y_mask))
    y_pred = jnp.where(labels == 0, 0, jnp.take(y_pred, labels, axis=-1))
    count = jnp.count_nonzero(y_pred)
    return -jnp.sum(y_pred) / count


lr = config.getfloat("training", "lr")
lr = optax.warmup_cosine_decay_schedule(lr, 0.1, 200, 2000, 0.5)
optimizer = optax.adam(learning_rate=lr)

opt_state, step = optim(
    model, optimizer, loss, in_axes=(None, 0, 0, 0, 0, 0), out_axes=0
)

model_path = path.join(config.get("training", "checkpoint_path"), "model.eqx")
opt_state_path = path.join(config.get("training", "checkpoint_path"), "opt_state.eqx")
if path.isfile(model_path) and path.isfile(opt_state_path):
    model = eqx.tree_deserialise_leaves(model_path, model)
    opt_state = eqx.tree_deserialise_leaves(opt_state_path, opt_state)

if config.getboolean("training", "wandb"):
    wandb_config = dict(config["model"])
    wandb_config["epochs"] = config.getint("training", "epochs")
    wandb_config["dataset"] = config.get("training", "dataset")

    run = wandb.init(
        project=config.get("training", "wandb_project"),
        notes=config.get("training", "wandb_notes"),
        config=wandb_config,
    )

print("Running...")
for e in range(config.getint("training", "epochs")):
    total_loss = 0
    num_batches = 0
    for i, (Xbt, ybt, labelbt) in enumerate(dataloader):
        Xbt, ybt, labelbt = [jnp.array(x) for x in (Xbt, ybt, labelbt)]
        Xmask, ymask = [text.create_pad_masks(x) for x in (Xbt, ybt)]

        model, opt_state, batch_loss = step(
            model, opt_state, Xbt, ybt, Xmask, ymask, labelbt
        )
        total_loss += batch_loss
        num_batches += 1

        if num_batches % config.getint("training", "checkpoint_freq") == 0:
            eqx.tree_serialise_leaves(model_path, model)
            eqx.tree_deserialise_leaves(opt_state_path, opt_state)
            print(
                f"Batches trained: {num_batches} | Avg. Batch loss: {total_loss/num_batches}"
            )

        if config.getboolean("training", "wandb"):
            wandb.log({"loss": total_loss / num_batches})

    epoch_loss = total_loss / num_batches
    print(f"Epoch {e} | loss: {epoch_loss}")