import requests

from dataloader.dataloaders import dataloader


def test_dataloader():
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

    # print(qa_dicts[:3])
    # assert 1 == 0