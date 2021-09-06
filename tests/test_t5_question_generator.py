import torch

from models.questiongenerator import T5QuestionGenerator


class TestT5QuestionGenerator:

    model = T5QuestionGenerator(model_id='t5-small')

    def test_batch_forward(self):
        answer_contexts = [['15 sticks', 'I picked up 15 sticks.'],
                           ['the tree', 'I climbed the tree one afternoon.']]
        questions = ['What did you pick up?', 'What did you climb?']
        with torch.no_grad():
            output = self.model(answer_contexts, questions=questions)
        assert output.logits.shape[0] == 2
