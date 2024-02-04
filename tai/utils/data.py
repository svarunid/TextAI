import os
from functools import partial

import jax
import tensorflow as tf
from clu import data
from etils.epath import Path
from tensorflow_text import SentencepieceTokenizer

AUTOTUNE = tf.data.AUTOTUNE


def create_tokenizer(config):
    """
    Creates a tokenizer from the given configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: Tuple of source and target language tokenizers.
    """

    def load_tokenizer(lang, alpha=1):
        with open(config["path"] / (lang + ".model"), "rb") as f:
            return SentencepieceTokenizer(f.read(), alpha=alpha)

    if config["use_separate_tokenizer"]:
        src_tok = load_tokenizer(config["src_lang"], alpha=config["alpha"])
        tgt_tok = load_tokenizer(config["tgt_lang"])
        return src_tok, tgt_tok

    tok = load_tokenizer(config["lang"], alpha=config["alpha"])
    return tok, tok


def create_dataset(root_dir, config, tok_config):
    """
    Creates a tf.data.Dataset from the given configuration. Automatically
    pads, repeats, caches and prefetches the dataset.

    Args:
        root_dir (Path): Path to the root directory.
        config (dict): Configuration dictionary.
        tok_config (dict): Tokenizer configuration dictionary.

    Returns:
        tf.data.Dataset: Dataset.
    """
    tok_config["path"] = root_dir / tok_config["path"]

    cache_dir = root_dir / config["cache_dir"]
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)

    src = root_dir / config["path"] / config["src"]
    tgt = root_dir / config["path"] / config["tgt"]

    map_with_autotune = lambda ds: partial(ds.map, num_parallel_calls=AUTOTUNE)
    src_ds, tgt_ds = [tf.data.TextLineDataset(lang) for lang in (src, tgt)]
    src_ds, tgt_ds = [
        map_with_autotune(ds)(tok.tokenize)
        for ds, tok in zip((src_ds, tgt_ds), create_tokenizer(tok_config))
    ]

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
        .cache(filename=os.fspath(cache_dir))
        .repeat(config["epochs"])
        .prefetch(AUTOTUNE)
    )

    return ds


class TfDatasetIterator(data.DatasetIterator):
    """
    A wrapper around tf.data.Dataset that implements the clu.data.DatasetIterator
    interface.
    """

    def __init__(self, dataset, checkpoint_dir) -> None:
        """
        Initializes the iterator.

        Args:
            dataset (tf.data.Dataset): Dataset to wrap.
            checkpoint_dir (Path): Path to the checkpoint directory.
        """
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
        """
        Resets the iterator.
        """
        self.iterator = iter(self._dataset)
        self._checkpointer = tf.train.Checkpoint(ds=self.iterator)

    def save(self, filename: Path):
        """
        Saves the iterator state to disk.

        Args:
            filename (Path): Path to save the iterator state to.
        """
        self._checkpointer.write(file_prefix=os.fspath(self.checkpoint_dir / filename))

    def restore(self, filename: Path):
        """
        Restores the iterator state from disk.

        Args:
            filename (Path): Path to restore the iterator state from.
        """
        self._checkpointer.read(os.fspath(self.checkpoint_dir / filename))
