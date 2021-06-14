import json
import os
import pathlib
import sys

sys.path.insert(0, '../code2vec')

import torch
import numpy as np

from code2vec import load_model_dynamically
from config import Config
from interactive_predict import InteractivePredictor
from model import PredictionHead

model = PredictionHead(384, 128, 128, 4)
model.load_state_dict(torch.load('./state_dict.uu', map_location=torch.device('cpu')))
model.eval()

config = Config(set_defaults=True, load_from_args=True, verify=True)
code2vec_model = load_model_dynamically(config)
predictor = InteractivePredictor(config, code2vec_model)

label2name = {
    0: 'parallel_inheritance_hierarchies',
    1: 'god_classes',
    2: 'data_class',
    3: 'feature_envy',
}


def get_embedding(filename):
    return torch.FloatTensor(
        np.array([float(x) for x in filename.open().read().split()]).reshape(384, -1)
    ).unsqueeze(0)


def predict():
    file = open(sys.argv[1]).read()
    filename = sys.argv[1].split('/')[-1]

    open(f'../code2vec/input/{filename}', 'w').write(file)

    predictor.predict('../code2vec/input', '../code2vec/output')
    embedding = get_embedding(
        pathlib.Path(f'../code2vec/output/{filename.split(".")[0]}' + '.txt')
    )
    probabilities = [
        torch.sigmoid(value).cpu().data.numpy().tolist() for value in model.forward(embedding)
    ][0]

    os.path.exists(f'../code2vec/input/{filename}') and os.remove(
        f'../code2vec/input/{filename}'
    )
    os.path.exists(f'../code2vec/output/{filename.split(".")[0]}.txt') and os.remove(
        f'../code2vec/output/{filename.split(".")[0]}.txt'
    )

    probs = {label2name[index]: value for index, value in enumerate(probabilities)}

    print(f'{filename}:')
    print(json.dumps(probs, indent=4))


if __name__ == '__main__':
    predict()
