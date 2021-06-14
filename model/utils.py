import os
import pathlib
from collections import defaultdict

import numpy as np


def get_antipattern_dict(
    embedding_dir, ignore_dirs=[pathlib.Path('./embeddings/c_plus_plus/negative_samples')]
):
    antipatterns = list(embedding_dir.glob('*'))
    antipatterns_dict = {}
    for antipattern in antipatterns:
        if antipattern not in ignore_dirs:
            antipatterns_dict[antipattern.name] = list(antipattern.glob('*'))
    return antipatterns_dict


def get_labels(antipatterns_dict):
    labels = defaultdict(lambda: [False, False, False, False, ''])
    name2label = {name: i for name, i in zip(antipatterns_dict, range(len(antipatterns_dict)))}
    label2name = {value: key for key, value in name2label.items()}
    for name, paths in antipatterns_dict.items():
        for path in paths:
            labels[os.path.basename(path)][name2label[name]] = True
            labels[os.path.basename(path)][-1] = path
    return labels, name2label, label2name


def get_embedding(filename):
    return np.array([float(x) for x in filename.open().read().split()])


def get_embeddings(labels):
    for name, values in labels.items():
        labels[name].append(get_embedding(values[-1]).reshape(384, -1))
    return labels
