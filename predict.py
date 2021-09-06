import json

import torch

from models.retrievalmodel import LongQAModel


def predict(question, model):
    start_logits, end_logits, input_ids, relevance_logits = model(question)
    topk_relevant = torch.topk(relevance_logits, k=relevance_logits.shape[0] // 2, dim=-1)
    starts = torch.argmax(start_logits, dim=-1)
    ends = torch.argmax(end_logits, dim=-1)
    best_answers = []
    for index in topk_relevant.indices:
        question_context = input_ids[index]
        start = starts[index]
        end = ends[index]
        best_answers.append(model.r_tokenizer.decode(question_context[start:end + 1]))

    best_answers = [(b[0].upper() + b[1:]) if len(b) > 1 else b.upper() for b in best_answers]
    return best_answers


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('contexts.json') as f:
        contexts = json.load(f)
    model = LongQAModel(contexts=contexts)
    model.load_state_dict(torch.load('modeldata/model_59_valid_accuracy=0.4741_valid_loss=2.7002.pt'))
    model = model.to(device)
    model.eval()
    sentences = [
        'To Send information into a Google Sheet get started',
        'How do I send information into a Google sheet to get started?'
    ]
    with torch.no_grad():
        for sentence in sentences:
            print(predict(sentence, model))
