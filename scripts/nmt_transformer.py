import optax
import wandb
import argparse
from os import path
import equinox as eqx
from jax.tree_util import tree_map

from nn.transformers import Transformer

argparser = argparse.ArgumentParser(
    description="A script to train a transformer model on a parallel corpus for machine translation tasks."
)

argparser.add_argument("--data", type=str, default=None, help="Path of the tokenized parallel corpus file to load.")
argparser.add_argument("--src", type=str, default=None, help="Path of the tokenized source language file to load.")
argparser.add_argument("--tgt", type=str, default=None, help="Path of the tokenized target language file to load.")
argparser.add_argument("-c", "--checkpoint", type=str, default=None, help="Path of the checkpoint to load.")
argparser.add_argument("-s", "--isstream", action="store_true", help="Whether the data is a stream or not.")
argparser.add_argument("-i", "--init", action="store_true", help="Whether to initialize the model or not.")
argparser.add_argument("-d", "--dim", type=int, default=512, help="Dimension of the model.")
argparser.add_argument("-e", "--emb", type=int, default=256, help="Size of the embedding layer.")
argparser.add_argument("-w", "--wandb", action="store_true", help="Whether to use wandb or not.")
argparser.add_argument("-l", "--lr", type=float, default=1e-3, help="Learning rate.")
argparser.add_argument("--invocab", type=int, default=5_000, help="Input vocabulary size.")
argparser.add_argument("--outvocab", type=int, default=5_000, help="Output vocabulary size.")
argparser.add_argument("--inseq", type=int, default=64, help="Input sequence length.")
argparser.add_argument("--outseq", type=int, default=64, help="Output sequence length.")
argparser.add_argument("--enc", type=int, default=3, help="Number of encoder layers.")
argparser.add_argument("--dec", type=int, default=3, help="Number of decoder layers.")
argparser.add_argument("--heads", type=int, default=8, help="Number of attention heads.")
argparser.add_argument("--batch", type=int, default=32, help="Batch size.")
argparser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")

args = argparser.parse_args()