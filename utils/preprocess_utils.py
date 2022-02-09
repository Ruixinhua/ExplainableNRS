import re
import string
from typing import Union

import torch
import numpy as np
from config.default_config import default_values


def word_tokenize(sent):
    """ Split sentence into word list using regex.
    Args:
        sent (str): Input sentence

    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


def text2index(text, word_dict, method="keep", ignore=True):
    # TODO: tokenize for compare
    return word2index(word_dict, word_tokenize(text), ignore)


def clean_text(text):
    rule = string.punctuation + "0123456789"
    return re.sub(rf'([^{rule}a-zA-Z ])', r" ", text)


def aggressive_process(text):
    from nltk.corpus import stopwords as stop_words
    stopwords = set(stop_words.words("english"))
    text = text.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    text = text.translate(str.maketrans("0123456789", ' ' * len("0123456789")))
    text = [w for w in text.split() if len(w) > 0 and w not in stopwords]
    return text


def tokenize(text, method="keep_all"):
    tokens = []
    text = clean_text(text)
    rule = string.punctuation + "0123456789"
    from torchtext.data import get_tokenizer
    tokenizer = get_tokenizer('basic_english')
    if method == "keep_all":
        tokens = tokenizer(re.sub(rf'([{rule}])', r" \1 ", text.lower()))
    elif method == "aggressive":
        tokens = aggressive_process(text)
    elif method == "alphabet_only":
        tokens = tokenizer(re.sub(rf'([{rule}])', r" ", text.lower()))
    return tokens


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
        self.process_method = kwargs.get("process_method", "keep_all")
        if self.embedding_type == "elmo":
            # TODO: need to fix for elmo embeddings
            from allennlp.modules.elmo import batch_to_ids
            self.tokenize = batch_to_ids
        elif self.embedding_type == "glove":
            self.word_dict = kwargs.get("word_dict", {})  # load dictionary for glove embedding
            self.ignore = kwargs.get("ignore", True)  # default skip unknown words
            self.tokenize = self.text2token
            self.pad_id = 0
        elif self.embedding_type in default_values["bert_embedding"]:
            from transformers import AutoTokenizer
            self.word_dict = AutoTokenizer.from_pretrained(self.embedding_type)
            self.tokenize = self.encode
            if self.embedding_type == "transfo-xl-wt103":
                self.word_dict.pad_token = self.word_dict.eos_token
            self.pad_id = self.word_dict.pad_token_id

    def encode(self, x: Union[str, list], max_length: int, return_tensors=True):
        # TODO: tokenize list input for transformer-based embedding and mask
        x_encoded = self.word_dict.encode(x, add_special_tokens=True, max_length=max_length, truncation=True)
        # mask = pad_sentence(np.ones_like(x_encoded).tolist(), max_length)
        x_padded = pad_sentence(x_encoded, max_length, self.pad_id)

        if return_tensors:
            x_padded = torch.tensor(x_padded, dtype=torch.long)
            # mask = torch.tensor(mask, dtype=torch.int8)
        return x_padded

    def text2token(self, x: Union[str, list], max_length: int, return_tensors=True):
        if isinstance(x, list):
            x_token = [text2index(_, self.word_dict, self.process_method, self.ignore) for _ in x]
            x_padded = np.concatenate([pad_sentence(_, max_length) for _ in x_token])
        else:
            x_padded = pad_sentence(text2index(x, self.word_dict, self.process_method, self.ignore), max_length)
        if return_tensors:
            x_padded = torch.tensor(x_padded, dtype=torch.long)
        return x_padded


def filter_tokens(docs, no_below=20, no_above=0.5):
    # Create a dictionary representation of the documents.
    from gensim.corpora import Dictionary
    dictionary = Dictionary(docs)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    return dictionary


def lemmatize(docs):
    from nltk.stem.wordnet import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [[lemmatizer.lemmatize(t) for t in d] for d in docs]


def add_bigram(docs, min_count=200):
    # Add bi-grams and trigrams to docs (only ones that appear 20 times or more).
    from gensim.models import Phrases
    bigram = Phrases(docs, min_count=min_count)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bi-gram, add to document.
                docs[idx].append(token)
    return docs


def get_bow_corpus(docs, dictionary):
    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(d) for d in docs]
    return corpus
