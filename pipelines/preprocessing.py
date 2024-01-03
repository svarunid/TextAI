import configparser
from os import path
from pathlib import Path

from helpers import spickle, text, nmt

config_dir = (
    Path(path.dirname(path.realpath(__file__))).parent
    / "config"
    / "preprocessing.ini"
)
config = configparser.ConfigParser()
config.read(config_dir)

task = config.get("preprocessing", "task")

if task == "seq2se2":
    