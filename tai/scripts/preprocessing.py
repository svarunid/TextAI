import os
from functools import partial
from pathlib import Path

import sentencepiece as spm
import yaml

root_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
config_dir = root_dir / "config" / "preprocessing.yaml"
with open(config_dir, "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)

data_path = Path(config["path"])

task = config["task"]
task_config = config[task]


def train_tokenizer(dir, vocab_size, control_symbols=[]):
    spm.SentencePieceTrainer.train(
        input=os.fspath(root_dir / data_path / dir),
        model_prefix=os.fspath(root_dir / data_path / dir.split(".")[0]),
        vocab_size=vocab_size,
        control_symbols=control_symbols,
        model_type=config["model"],
        normalization_rule_name=config["norm_rule"],
        shuffle_input_sentence=config["shuffle"],
        input_sentence_size=config["sample_size"],
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        unk_surface="<unk>",
    )


tok_path = Path("tokenizer").joinpath(*data_path.parts[1:])

if not tok_path.exists():
    tok_path.mkdir(parents=True, exist_ok=True)

match task:
    case "seq2seq":
        train_tokenizer(task_config["src"], task_config["src_vocab"])
        train_tokenizer(task_config["tgt"], task_config["tgt_vocab"])
    case "mlm":
        train_tokenizer(
            task_config["src"],
            task_config["vocab"],
            control_symbols=[task_config["cls_token"], task_config["sep_token"]],
        )
