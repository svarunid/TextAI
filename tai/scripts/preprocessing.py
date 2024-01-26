from functools import partial
from os import makedirs, path
from pathlib import Path

import sentencepiece as spm
import yaml

root_dir = Path(path.dirname(path.realpath(__file__))).parent.parent
config_dir = root_dir / "config" / "preprocessing.yaml"
with open(config_dir, "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)

input_type = config["input_type"]
data_path = Path(config["path"])

if input_type == "text":
    data_config = config[input_type]

    task = data_config["task"]
    task_config = data_config[task]

    train_tokenizer = partial(
        spm.SentencePieceTrainer.train,
        model_type=data_config["model"],
        normalization_rule_name=data_config["norm_rule"],
        shuffle_input_sentence=data_config["shuffle"],
        input_sentence_size=data_config["sample_size"],
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        unk_surface="<unk>",
    )

    tok_path = Path("tokenizer").joinpath(*data_path.parts[1:])

    if not tok_path.exists():
        makedirs(tok_path, exist_ok=True)

    if task == "seq2seq":
        train_tokenizer(
            input=str((root_dir / data_path / task_config["src"])),
            model_prefix=str((root_dir / tok_path / task_config["src"].split(".")[0])),
            vocab_size=task_config["src_vocab"],
        )
        train_tokenizer(
            input=str((root_dir / data_path / task_config["tgt"])),
            model_prefix=str((root_dir / tok_path / task_config["tgt"].split(".")[0])),
            vocab_size=task_config["tgt_vocab"],
        )
