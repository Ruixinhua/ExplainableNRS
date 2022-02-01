import re
import string

from gensim.corpora import Dictionary
from gensim.models import Phrases
from nltk.corpus import stopwords as stop_words
from nltk.stem.wordnet import WordNetLemmatizer
from torchtext.data import get_tokenizer


def text2index(text, word_dict, method="keep", ignore=True):
    return word2index(word_dict, tokenize(text, method), ignore)


def clean_text(text):
    rule = string.punctuation + "0123456789"
    return re.sub(rf'([^{rule}a-zA-Z ])', r" ", text)


def aggressive_process(text):
    stopwords = set(stop_words.words("english"))
    text = text.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    text = text.translate(str.maketrans("0123456789", ' ' * len("0123456789")))
    text = [w for w in text.split() if len(w) > 0 and w not in stopwords]
    return text


def tokenize(text, method="keep_all"):
    tokens = []
    text = clean_text(text)
    rule = string.punctuation + "0123456789"
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


def filter_tokens(docs, no_below=20, no_above=0.5):
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    return dictionary


def lemmatize(docs):
    lemmatizer = WordNetLemmatizer()
    return [[lemmatizer.lemmatize(t) for t in d] for d in docs]


def add_bigram(docs, min_count=200):
    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(docs, min_count=min_count)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
    return docs


def get_bow_corpus(docs, dictionary):
    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(d) for d in docs]
    return corpus
