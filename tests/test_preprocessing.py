from transformers import BertTokenizerFast

from utils.preprocessing import convert_str_indices_to_token_indices


class TestStringToTokens:
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def test_question_answer_str_to_tokens(self):
        question_context = [['How many toys are on the floor?',
                             'The floor was covered in toys. 15 of them to be exact.']]
        start_token, end_token = convert_str_indices_to_token_indices(question_context,
                                                                      31,
                                                                      33,
                                                                      self.tokenizer)
        encoded_question_context = self.tokenizer(question_context, add_special_tokens=True)['input_ids']
        answer_tokens = encoded_question_context[0][start_token:end_token + 1]
        decoded_answer = self.tokenizer.decode(answer_tokens)
        assert decoded_answer == '15'

    def test_sequence_str_to_tokens(self):
        sequence = ['The floor was covered in toys. 15 of them to be exact.']
        start_token, end_token = convert_str_indices_to_token_indices(sequence,
                                                                      31,
                                                                      33,
                                                                      self.tokenizer)
        encoded_question_context = self.tokenizer(sequence, add_special_tokens=True)['input_ids']
        answer_tokens = encoded_question_context[0][start_token:end_token + 1]
        decoded_answer = self.tokenizer.decode(answer_tokens)
        assert decoded_answer == '15'
