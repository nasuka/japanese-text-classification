import os
import re
import yaml
import MeCab


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f)
    return config

BASE_PATH = os.getcwd() + '/../'
CONFIG_PATH = BASE_PATH + 'config.yaml'

CONFIG = load_config(CONFIG_PATH)
TAGGER = MeCab.Tagger(CONFIG['MECAB_DICT'])
WORD_SPLITTER = re.compile(" ")


def tokenize(doc):
    result = TAGGER.parse(doc)
    words = [word for word in WORD_SPLITTER.split(result)]
    if words[-1] == "\n":
        words = words[:-1]
    return words