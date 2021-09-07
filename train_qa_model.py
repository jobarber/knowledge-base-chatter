import requests

from dataloader.dataloaders import question_answer_dataloader
from models.retrievalmodel import LongQAModel
from trainer.trainers import Trainer


def train_qa_model(epochs=5):
    json_response = requests.get(
        'https://www.dropbox.com/s/uge6kufl37x77h0/squad_formatted_train_20210704.json?dl=1').json()
    qa_dicts = []
    paragraphs = json_response['data'][0]['paragraphs']
    for paragraph in paragraphs[:15]:
        context = paragraph['context']
        qas = paragraph['qas']
        for qa in qas:
            answer = qa['answers'][0]
            if answer['answer_start'] >= 2000:
                continue
            sample = dict(context=context, answer=answer['text'], question=qa['question'])
            qa_dicts.append(sample)
    model = LongQAModel(contexts=list(set(d['context'][:2000] for d in qa_dicts)))

    # get data loaders
    train_dataloader = question_answer_dataloader(qa_dicts,
                                                  fast_tokenizer=model.r_tokenizer,
                                                  split='train',
                                                  batch_size=32,
                                                  train_size=0.9,
                                                  shuffle=True)
    valid_dataloader = question_answer_dataloader(qa_dicts,
                                                  fast_tokenizer=model.r_tokenizer,
                                                  split='valid',
                                                  batch_size=32,
                                                  train_size=0.9,
                                                  shuffle=True)
    # train model
    trainer = Trainer(model,
                      submodule_to_train='r_model',
                      tokenizer=model.r_tokenizer,
                      dataloader=train_dataloader,
                      validation_dataloader=valid_dataloader,
                      lr=1e-5,
                      epochs=epochs,
                      weight_decay=3e-1)
    trainer.train()


if __name__ == '__main__':
    train_qa_model()
