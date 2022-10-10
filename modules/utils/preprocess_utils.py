import re
from typing import Union

import torch
import numpy as np
import config.default_config as default_config


def word_tokenize(sent, method="keep_all"):
    """ Split sentence into word list using regex.
    Args:
        sent (str): Input sentence
        method (str): Tokenize method, keep_all or use_tokenize.
    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        if method == "keep_all":
            return pat.findall(sent.lower())
        elif method == "use_tokenize":
            return sent.lower().split()
        return pat.findall(sent.lower())
    else:
        return []


def text2index(text, word_dict, method="keep_all", ignore=True):
    # TODO: tokenize for compare
    return word2index(word_dict, word_tokenize(text, method), ignore)


def word2index(word_dict, sent, ignore=True):
    word_index = []
    for word in sent:
        if ignore:
            index = word_dict[word] if word in word_dict else 0
        else:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
            index = word_dict[word]
        word_index.append(index)
    return word_index


def pad_sentence(x, max_length, pad_id=0):
    return x[:max_length] + [pad_id for _ in range(max(0, max_length - len(x)))]


class Tokenizer:
    def __init__(self, **kwargs):
        self.embedding_type = kwargs.get("embedding_type", "glove")
        self.tokenized_method = kwargs.get("tokenized_method", "keep_all")
        if self.embedding_type == "elmo":
            # TODO: Not implemented for elmo
            from allennlp.modules.elmo import batch_to_ids
            self.tokenize = batch_to_ids
            self.pad_id = 0
        elif self.embedding_type == "glove":
            from modules.utils.dataset_utils import load_word_dict
            self.word_dict = kwargs.get("word_dict", load_word_dict(**kwargs))  # load dictionary for glove embedding
            self.ignore = kwargs.get("ignore", True)  # default skip unknown words
            self.tokenize = self.text2token
            self.pad_id = 0
        elif self.embedding_type in default_config.TEST_CONFIGS["bert_embedding"]:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_type)
            if self.embedding_type == "transfo-xl-wt103":
                self.word_dict = self.tokenizer.sym2idx
            else:
                self.word_dict = self.tokenizer.vocab
            self.tokenize = self.encode
            if self.embedding_type == "transfo-xl-wt103":
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.pad_id = self.tokenizer.pad_token_id

    def encode(self, x: Union[str, list], max_length: int, return_tensors=True):
        # TODO: tokenize list input for transformer-based embedding and mask
        x_encoded = self.tokenizer(x, add_special_tokens=True, max_length=max_length, truncation=True, padding=True)
        # mask = pad_sentence(np.ones_like(x_encoded).tolist(), max_length)
        x_padded = x_encoded["input_ids"]

        if return_tensors:
            x_padded = torch.tensor(x_padded, dtype=torch.long)
            # mask = torch.tensor(mask, dtype=torch.int8)
        return x_padded

    def text2token(self, x: Union[str, list], max_length: int, return_tensors=True):
        if isinstance(x, list):
            x_token = [text2index(_, self.word_dict, self.tokenized_method, self.ignore) for _ in x]
            x_padded = np.array([pad_sentence(_, max_length) for _ in x_token])
        else:
            x_padded = pad_sentence(text2index(x, self.word_dict, self.tokenized_method, self.ignore), max_length)
        if return_tensors:
            x_padded = torch.tensor(x_padded, dtype=torch.long)
        return x_padded
