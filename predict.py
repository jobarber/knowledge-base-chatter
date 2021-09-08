import torch


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
