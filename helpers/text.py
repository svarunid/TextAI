import unicodedata
from itertools import chain, islice, repeat

import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece as spm


def train_spm(iterable, prefix, vocab=2_000, sentence_size=50_000, model_type="bpe"):
    """
    Train a sentence-piece tokenizer.
    The pad, unk, bos, eos tokens corresponds to 0, 1, 3 and 4 id respectively.

    Parameters
    ----------
    iterable: Sequence[str]
        List of sequence to train the tokenizer.
    prefix: str
        A prefix for .model, .vocab files.
    vocab: int, optional
        Size of the vocabulary (default is 2_000)
    sentence_size: int, optional
        Size of sentences to sample for training. (default is 50_000)
    model_type: str, optional
        Type of model. Either "bpe" or "unigram". (default is "bpe")
    """
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(iterable),
        model_prefix=prefix,
        vocab_size=vocab,
        model_type="bpe",
        normalization_rule_name="identity",
        input_sentence_size=sentence_size,
        shuffle_input_sentence=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        unk_surface="<unk>",
    )


def load_spm(path):
    """
    Load a pretrained tokenizer

    Parameters
    ----------
    path: str
        Path to the model file

    Returns
    -------
    SentencePieceProcessor
        A trained sentence-piece tokenizer.
    """
    local_spm = spm.SentencePieceProcessor(model_file=f"{path}.model")
    return local_spm


def normalize(arr, form):
    """
    Performs unicode noramlization on sentences.

    Paramters
    ---------
    arr: str
        A list of pairs of sentences.
    form: str
        Normalization form to use. NFD/NFC/NFKD/NFKC. (default is "NFC")

    Returns
    -------
    list
        A list of normalized sentences.
    """
    return [unicodedata.normalize(form, seq) for seq in arr]


def create_pad_masks(arr: jax.Array, pad=0):
    """
    Creates a mask array in the same shape as the given a padded array.

    Parameters
    ----------
    arr: jax.Array
        Padded Array
    pad: int, optional
        The ID to use for padding. (defualt is 0)
    mode: str, optional
        Mode of the masking operation. Either "additive" or "multiplicative". (default is "multiplicative")

    Returns
    -------
    jax.Array
        A mask array.
    """
    return jnp.where(arr==0, jnp.full_like(arr, jnp.NINF, dtype=arr.dtype), 0)


def pad_or_truncate(seq, size=32, pad=0):
    """
    Pads and truncates a list tokens a specific length.

    Parameters:
    seq: list
        A list of tokens.
    size: int, optional
        Max size of the sequence. (default is 32)
    pad: int, optional
        The ID to use for padding. (defualt is 0)

    Returns
    -------
    list
        Truncated or padded list of tokens.
    """
    return list(islice(chain(seq, repeat(pad)), size))


def seq2seq_batched_iterator(data, in_seq, out_seq, batch_size=32):
    """
    A dataloader for sequence to sequence tasks.
    Loads data in batches.

    Parameters
    ----------
    data: Iterator
        Pairs of Sequences.
    in_seq: int
        Input sequence length.
    out_seq: int
        Output sequence length.
    batch_size: int, optional
        Size of a batch. (default is 32)

    Yields
    ------
    Xbt: list
        Batch of source sequences.
    ybt: list
        Batch of target sequences.
    labelbt: list
        Batch of labels for target sequences.
    """
    while batch := tuple(islice(data, batch_size)):
        X = [x[0] for x in batch]
        y = [x[1] for x in batch]

        Xbt = [pad_or_truncate(seq, in_seq) for seq in X]
        y = [pad_or_truncate(seq, out_seq + 1) for seq in y]
        ybt = [x[:-1] for x in y]
        labelbt = [x[1:] for x in y]

        yield Xbt, ybt, labelbt

def subsequent_mask(size):
    """
    Creates a mask for the target sequence.
    Used to prevent the model from atttening to future tokens.

    Parameters
    ----------
    size: int
        Size of the mask.

    Returns
    -------
    jax.Array
        A mask array.
    """
    arr = jnp.triu(jnp.ones((size, size), dtype=jnp.int16), 1) == 1
    return jnp.where(arr, jnp.full_like(arr, jnp.NINF, dtype=arr.dtype), 0)
