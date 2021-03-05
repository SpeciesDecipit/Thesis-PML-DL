import os
import pathlib

import numpy as np
import torch
from flask import Flask, jsonify, request

from web_service.model import PredictionHead

app = Flask(__name__)

model = PredictionHead(384, 128, 128, 4)
model.load_state_dict(torch.load('./state_dict'))
model.eval()


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
        os.system(
            'python ../code2vec/code2vec.py --load ../code2vec/models/java14_model/saved_model_iter8.release --predict --export_code_vectors --target_source_code ../code2vec/input --target_source_code_embeddings_output ../code2vec/output'
        )
        embedding = get_embedding(
            pathlib.Path(f'../code2vec/output/{file.filename.split(".")[0]}' + '.txt')
        )
        probabilities = [
            torch.sigmoid(value).cpu().data.numpy().tolist() for value in model.forward(embedding)
        ][0]
        return jsonify({label2name[index]: value for index, value in enumerate(probabilities)})


if __name__ == '__main__':
    app.run()
