import json

import torch

from models.retrievalmodel import LongQAModel
from predict import predict


def test_predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('contexts.json') as f:
        contexts = json.load(f)
    model = LongQAModel(contexts=contexts)
    model = model.to(device)
    model.eval()
    sentences = [
        'How do I change my password?',
        'How do I send information into a Google sheet to get started?'
    ]
    with torch.no_grad():
        for sentence in sentences:
            print(predict(sentence, model))
    assert False
