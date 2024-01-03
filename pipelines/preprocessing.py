import configparser
from os import path, makedirs
from pathlib import Path
import pickle

from helpers import spickle, text, nmt

config_dir = (
    Path(path.dirname(path.realpath(__file__))).parent
    / "config"
    / "preprocessing.ini"
)
config = configparser.ConfigParser()
config.read(config_dir)

task = config.get("preprocessing", "task")
path = config.get(task, "data_path")
shuffle = config.getboolean("preprocessing", "shuffle")
train_size = config.getfloat("preprocessing", "train_size")
dev_size = config.getfloat("preprocessing", "dev_size")

if task == "seq2seq":
    src_path, tgt_path = config.get(task, "src"), config.get(task, "tgt")
    src, tgt = nmt.load_parallel_corpus(f"{path}/{src_path}", f"{path}/{tgt_path}")
    if config.getboolean(task, "normalize"):
        src, tgt = text.normalize(src), text.normalize(tgt)

Xtr, Xdev, Xte = text.split(src, train_size, dev_size, shuffle)
ytr, ydev, yte = text.split(tgt, train_size, dev_size, shuffle)
del src, tgt

makedirs(config.get("tokenizer", "path"), exist_ok=True)

text.train_spm(
    src, 
    config.get("tokenizer", "path") + "src",
    config.getint(task, "vocab"),
    config.getint("tokenizer", "sample_size"),
)
text.train_spm(
    tgt, 
    config.get("tokenizer", "path") + "tgt",
    config.getint(task, "vocab"),
    config.getint("tokenizer", "sample_size"),
)

src_spm, tgt_spm = (
    text.load_spm(config.get("tokenizer", "path") + "src"),
    text.load_spm(config.get("tokenizer", "path") + "tgt")
)

Xtr, Xdev, Xte = (
    src_spm.tokenize(Xtr, alpha=config.getfloat("tokenizer", "alpha")), 
    src_spm.tokenize(Xdev), 
    src_spm.tokenize(Xte)
)
ytr, ydev, yte = (
    tgt_spm.tokenize(ytr, add_bos=True, add_eos=True), 
    tgt_spm.tokenize(ydev, add_bos=True, add_eos=True), 
    tgt_spm.tokenize(yte, add_bos=True, add_eos=True)
)

makedirs(config.get("tokenizer", "save_path"), exist_ok=True)

spickle.sdump(config.get("tokenizer", "save_path") + "/src.spkl", zip(Xtr, ytr))
pickle.dump(zip(Xdev, ydev), open(config.get("tokenizer", "save_path") + "/dev.pkl", "wb"))
pickle.dump(zip(Xte, yte), open(config.get("tokenizer", "save_path") + "/test.pkl", "wb"))