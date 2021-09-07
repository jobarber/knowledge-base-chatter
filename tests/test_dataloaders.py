import requests
from transformers import BertTokenizerFast

from dataloader.dataloaders import question_answer_dataloader, question_generator_dataloader


def test_question_answer_dataloader():
    json_response = requests.get(
        'https://www.dropbox.com/s/uge6kufl37x77h0/squad_formatted_train_20210704.json?dl=1').json()

    qa_dicts = []
    paragraphs = json_response['data'][0]['paragraphs']
    for paragraph in paragraphs:
        context = paragraph['context']
        qas = paragraph['qas']
        for qa in qas:
            answer = qa['answers'][0]
            sample = dict(context=context, answer=answer['text'], question=qa['question'])
            qa_dicts.append(sample)

    tokenizer = BertTokenizerFast.from_pretrained('bert-large-cased')
    dataloader = question_answer_dataloader(qa_dicts[:5], tokenizer, split='train', batch_size=2)
    for batch in dataloader:
        assert len(batch[0]) == 2
        break


def test_question_generator_dataloader():
    dataloader = question_generator_dataloader(split='train', batch_size=3)
    for batch in dataloader:
        assert len(batch['answers']['text'][0]) == 3
        break
