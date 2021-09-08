import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5QuestionGenerator(nn.Module):

    def __init__(self,
                 model_id='t5-large',
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(T5QuestionGenerator, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(model_id)
        self.device = device
        self.to(self.device)

    def forward(self,
                answer_contexts,
                questions=None):
        """

        Parameters
        ----------
        answer_contexts
        labels

        Returns
        -------

        """
        # Tokenize answer-contexts with special task formatting
        formatted_answer_contexts = []
        for answer, context in answer_contexts:
            answer_context = (f'answer: {answer} '
                              f'context: {context} '
                              f'question:{self.tokenizer.pad_token}')
            formatted_answer_contexts.append(answer_context)
        tokenized_answer_contexts = self.tokenizer(formatted_answer_contexts,
                                                   return_tensors='pt',
                                                   padding=True).to(self.device)

        # Tokenize questions--if they exist--as labels
        labels = None
        if questions:
            labels = self.tokenizer(questions,
                                    return_tensors='pt',
                                    padding=True).to(self.device)

        return self.model(input_ids=tokenized_answer_contexts['input_ids'],
                          labels=labels['input_ids'])
