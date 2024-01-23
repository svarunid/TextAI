import tensorflow as tf
from tensorflow_text import SentencepieceTokenizer

AUTOTUNE = tf.data.AUTOTUNE


def create_tokenizer(config, path):
    if config["use_separate_tokenizer"]:
        with open(
            path / f"{config['src_lang']}.model",
            "rb",
        ) as f:
            src_tok = SentencepieceTokenizer(f.read())
        with open(
            path / f"{config['tgt_lang']}.model",
            "rb",
        ) as f:
            tgt_tok = SentencepieceTokenizer(
                f.read(),
                alpha=config["alpha"],
            )
        return src_tok, tgt_tok
    with open(path / f"{config['lang']}.model", "rb") as f:
        tok = SentencepieceTokenizer(
            f.read(),
            alpha=config["alpha"],
        )
    return tok


def create_dataset(config, tok_config, src, tgt, one_hot=True):
    src_ds = tf.data.TextLineDataset(src)
    tgt_ds = tf.data.TextLineDataset(tgt)
    if tok_config["use_separate_tokenizer"]:
        src_tok, tgt_tok = create_tokenizer(tok_config, tok_config["path"])
        src_ds = src_ds.map(src_tok.tokenize, num_parallel_calls=AUTOTUNE)
        tgt_ds = tgt_ds.map(tgt_tok.tokenize, num_parallel_calls=AUTOTUNE)
    else:
        tok = create_tokenizer(tok_config, tok_config["path"])
        src_ds = src_ds.map(tok.tokenize, num_parallel_calls=AUTOTUNE)
        tgt_ds = tgt_ds.map(tok.tokenize, num_parallel_calls=AUTOTUNE)
    src_ds = src_ds.map(
        lambda x: x[: config["src_max_len"]], num_parallel_calls=AUTOTUNE
    )
    tgt_ds = tgt_ds.map(
        lambda x: tf.concat([[2], x[: config["tgt_max_len"]]], axis=-1),
        num_parallel_calls=AUTOTUNE,
    )
    labels_ds = tgt_ds.map(
        lambda x: tf.concat([x[1:], [3]], axis=-1), num_parallel_calls=AUTOTUNE
    )

    ds = (
        tf.data.Dataset.zip((src_ds, tgt_ds, labels_ds))
        .cache()
        .padded_batch(
            config["batch_size"],
            padded_shapes=(
                config["src_max_len"],
                config["tgt_max_len"],
                config["tgt_max_len"],
            ),
        )
        .repeat(config["epochs"])
        .prefetch(AUTOTUNE)
    )
    return ds
