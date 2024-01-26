import os
from functools import partial

import jax
import tensorflow as tf
from clu import data
from etils.epath import Path
from tensorflow_text import SentencepieceTokenizer

AUTOTUNE = tf.data.AUTOTUNE


def create_tokenizer(config):
    def load_tokenizer(lang, alpha=1):
        with open(config["path"] / (lang + ".model"), "rb") as f:
            return SentencepieceTokenizer(f.read(), alpha=alpha)

    if config["use_separate_tokenizer"]:
        src_tok = load_tokenizer(config["src_lang"])
        tgt_tok = load_tokenizer(config["tgt_lang"], alpha=config["alpha"])
        return src_tok, tgt_tok

    tok = load_tokenizer(config["lang"], alpha=config["alpha"])
    return tok, tok


def create_dataset(config, tok_config, src, tgt):
    map_with_autotune = lambda ds: partial(ds.map, num_parallel_calls=AUTOTUNE)
    to_dlpack = tf.experimental.dlpack.to_dlpack
    src_ds = tf.data.TextLineDataset(src)
    tgt_ds = tf.data.TextLineDataset(tgt)
    src_tok, tgt_tok = create_tokenizer(tok_config)
    src_ds = map_with_autotune(src_ds)(src_tok.tokenize)
    tgt_ds = map_with_autotune(tgt_ds)(tgt_tok.tokenize)
    src_ds = map_with_autotune(src_ds)(lambda x: x[: config["src_max_len"]])
    tgt_ds = map_with_autotune(tgt_ds)(
        lambda x: tf.concat([[2], x[: config["tgt_max_len"] - 1]], axis=-1)
    )
    labels_ds = map_with_autotune(tgt_ds)(
        lambda x: tf.concat([x[1:], [3]], axis=-1),
    )

    ds = (
        tf.data.Dataset.zip((src_ds, tgt_ds, labels_ds))
        .padded_batch(
            config["batch_size"],
            padded_shapes=(
                config["src_max_len"],
                config["tgt_max_len"],
                config["tgt_max_len"],
            ),
        )
        .cache(filename=str(config["cache_dir"]))
        .repeat(config["epochs"])
    )

    return ds


class TfDatasetIterator(data.DatasetIterator):
    def __init__(self, dataset, checkpoint_dir) -> None:
        super().__init__()
        self._dataset = dataset
        self.checkpoint_dir = Path(checkpoint_dir)
        self.iterator = iter(dataset)
        self._checkpointer = tf.train.Checkpoint(ds=self.iterator)

    @property
    def element_spec(self):
        elememt_spec = self._dataset.element_spec

        return tuple(
            data.ArraySpec(dtype=el.dtype.as_numpy_dtype, shape=tuple(el.shape))
            for el in elememt_spec
        )

    def __next__(self):
        to_dlpack = tf.experimental.dlpack.to_dlpack
        to_array = jax.dlpack.from_dlpack
        return tuple(to_array(to_dlpack(el)) for el in next(self.iterator))

    def reset(self):
        self.iterator = iter(self._dataset)
        self._checkpointer = tf.train.Checkpoint(ds=self.iterator)

    def save(self, filename: Path):
        self._checkpointer.write(file_prefix=os.fspath(self.checkpoint_dir / filename))

    def restore(self, filename: Path):
        self._checkpointer.read(os.fspath(self.checkpoint_dir / filename))
