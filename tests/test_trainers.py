import pytest
import requests
from sklearn import metrics

from dataloader.dataloaders import dataloader
from models.retrievalmodel import LongQAModel
from trainer.trainers import Trainer


@pytest.mark.skip(reason="this test takes a long time")
def test_dpr_trainer():
    json_response = requests.get(
        'https://www.dropbox.com/s/uge6kufl37x77h0/squad_formatted_train_20210704.json?dl=1').json()

    qa_dicts = []
    paragraphs = json_response['data'][0]['paragraphs']
    for paragraph in paragraphs[:20]:
        context = paragraph['context']
        qas = paragraph['qas']
        for qa in qas:
            answer = qa['answers'][0]
            if answer['answer_start'] >= 100:
                continue
            sample = dict(context=context, answer=answer['text'], question=qa['question'])
            qa_dicts.append(sample)

    # instantiate model
    model = LongQAModel(contexts=list(set(d['context'][:100] for d in qa_dicts)))

    # get data loader
    train_dataloader = dataloader(qa_dicts,
                                  fast_tokenizer=model.r_tokenizer,
                                  split='train',
                                  batch_size=32,
                                  train_size=0.9,
                                  shuffle=True)
    valid_dataloader = dataloader(qa_dicts,
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
                      epochs=4,
                      weight_decay=3e-1,
                      val_metrics={'accuracy': metrics.accuracy_score,
                                   'f1': metrics.f1_score,
                                   'precision': metrics.precision_score,
                                   'recall': metrics.recall_score})
    val_metrics = trainer.train()
    assert set(val_metrics.keys()) == set(['accuracy', 'f1', 'precision', 'recall', 'loss'])
