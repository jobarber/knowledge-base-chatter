import torch


def convert_str_indices_to_token_indices(text,
                                         start_str_index,
                                         end_str_index,
                                         fast_tokenizer,
                                         **tokenizer_kwargs):
    """
    Converts string indices common in QA tasks to token indices.
    Parameters
    ----------
    fast_tokenizer : instance of PreTrainedTokenizerFast
        We have to use a fast tokenizer in order to access offset mappings.
    text : str or list of (single) list of 2 strings (for
        2 sent tasks)
        The full text of the question and context or just the context,
        depending on the situation.
    start_end_str_indices : sequence with two integers.
        Contains start string index and end string index of the answer.
    tokenizer_kwargs : dict
        Any remaining keyword arguments.
    Example
    -------
    # >>> tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # >>> question_context = [['How many toys are on the floor?',
    #                          'The floor was covered in toys. 15 of them to be exact.']]
    # >>> convert_str_indices_to_token_indices(tokenizer,
    #                                          question_context,
    #                                          [62, 64])
    (16, 18)

    Returns
    -------
    A tuple with the start token index and end token index.
    """
    tokenized = fast_tokenizer(text,
                               return_offsets_mapping=True,
                               return_tensors='pt',
                               **tokenizer_kwargs)

    offset_mapping = tokenized['offset_mapping'].reshape(-1, 2)
    token_offsets = list(enumerate(offset_mapping))
    for token_index, offset in token_offsets:
        if offset[0] <= start_str_index <= offset[1]:
            start_token = token_index
        if offset[0] <= end_str_index <= offset[1]:
            end_token = token_index

    return start_token, end_token
