import pytest
import requests
from sklearn import metrics

from dataloader.dataloaders import question_answer_dataloader
from models.retrievalmodel import LongQAModel
from trainer.trainers import Trainer


class TestQATrainers:
    qa_dicts = [
        # quotations taken from Steve McConnell, Code Complete
        dict(context="Good code is its own best documentation. As you're about to add a comment, "
                     "ask yourself, 'How can I improve the code so that this comment isn't needed?' "
                     "Improve the code and then document it to make it even clearer.",
             question="What should I do after I improve my code?",
             answer="document it to make it even clearer"),
        dict(context="Trying to improve software quality by increasing the amount of testing is like "
                     "trying to lose weight by weighing yourself more often. What you eat before you "
                     "step onto the scale determines how much you will weigh, and the software-development "
                     "techniques you use determine how many errors testing will find.",
             question="What will software development techniques determine?",
             answer="how many errors testing will find"),
        dict(context="The road to programming hell is paved with global variables.",
             question="How would you characterize the road to programing hell?",
             answer="paved with global variables"),
        dict(context="Reduce complexity. The single most important reason to create a routine is to "
                     "reduce a program's complexity. Create a routine to hide information so that you "
                     "won't need to think about it.",
             question="What is the most important reason to create a routine?",
             answer="to reduce a program's complexity"),
    ]

    # @pytest.mark.skip(reason="this test takes a long time")
    def test_dpr_trainer(self):
        # instantiate model
        model = LongQAModel(contexts=[d['context'] for d in self.qa_dicts])

        # get data loader
        train_dataloader = question_answer_dataloader(self.qa_dicts,
                                                      fast_tokenizer=model.r_tokenizer,
                                                      split='train',
                                                      batch_size=32,
                                                      train_size=0.75,
                                                      shuffle=True)
        valid_dataloader = question_answer_dataloader(self.qa_dicts,
                                                      fast_tokenizer=model.r_tokenizer,
                                                      split='valid',
                                                      batch_size=32,
                                                      train_size=0.75,
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

        # @pytest.mark.skip(reason="this test takes a long time")
        def test_dpr_trainer(self):
            # instantiate model
            model = LongQAModel(contexts=[d['context'] for d in self.qa_dicts])

            # get data loader
            train_dataloader = question_answer_dataloader(self.qa_dicts,
                                                          fast_tokenizer=model.r_tokenizer,
                                                          split='train',
                                                          batch_size=32,
                                                          train_size=0.75,
                                                          shuffle=True)
            valid_dataloader = question_answer_dataloader(self.qa_dicts,
                                                          fast_tokenizer=model.r_tokenizer,
                                                          split='valid',
                                                          batch_size=32,
                                                          train_size=0.75,
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