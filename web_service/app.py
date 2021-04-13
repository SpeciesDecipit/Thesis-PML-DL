import sys

sys.path.insert(0, '../code2vec')

import os
import pathlib

import numpy as np
import torch
from flask import Flask, jsonify, request

from code2vec import load_model_dynamically
from config import Config
from interactive_predict import InteractivePredictor
from web_service.model import PredictionHead


app = Flask(__name__)

model = PredictionHead(384, 128, 128, 4)
model.load_state_dict(torch.load('./state_dict.uu', map_location=torch.device('cpu')))
model.eval()

config = Config(set_defaults=True, load_from_args=True, verify=True)
code2vec_model = load_model_dynamically(config)
predictor = InteractivePredictor(config, code2vec_model)
config.log('Done creating code2vec model')
# predictor.predict()


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


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']

        file.save(f'../code2vec/input/{file.filename}')
        predictor.predict('../code2vec/input', '../code2vec/output')
        embedding = get_embedding(
            pathlib.Path(f'../code2vec/output/{file.filename.split(".")[0]}' + '.txt')
        )
        probabilities = [
            torch.sigmoid(value).cpu().data.numpy().tolist() for value in model.forward(embedding)
        ][0]
        return jsonify({label2name[index]: value for index, value in enumerate(probabilities)})


if __name__ == '__main__':
    app.run()
