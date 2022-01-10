import logging
import os
from pathlib import Path

import numpy as np

from experiment import init_args, customer_args, ConfigParser
from utils import load_dataset_df, tokenize, get_project_root
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel


def lda_model(dictionary, corpus, **kwargs):
    temp = dictionary[0]  # This is only to "load" the dictionary.
    # Make an index to word dictionary.
    args = {
        "corpus": corpus, "id2word": dictionary.id2token, "chunksize": kwargs.get("chunksize", 2000),
        "alpha": kwargs.get("alpha", "auto"), "eta": kwargs.get("eta", "auto"), "passes": kwargs.get("passes", 10),
        "iterations": kwargs.get("iterations", 400), "num_topics": kwargs.get("num_topics", 50),
        "eval_every": kwargs.get("eval_every", None)
    }
    return LdaModel(**args)


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


def evaluate_topics(model, corpus, docs, dictionary, num_topics=50, c_method="c_npmi", topn=20):
    top_topics = model.top_topics(corpus, texts=docs, dictionary=dictionary, coherence=c_method, topn=topn)

    # Average topic coherence is the sum of topic coherence of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print(f'Average topic coherence({c_method}): %.4f.' % avg_topic_coherence)

    from pprint import pprint
    pprint(top_topics)


def load_docs(name, method):
    df, _ = load_dataset_df(name)
    docs = [tokenize(d, method) for d in df["data"].values]
    if do_lemma:
        docs = lemmatize(docs)
    if add_bi:
        docs = add_bigram(docs, config.get("min_count", 200))
    return docs


def save_topic_embed(model, dictionary, saved_file):
    topics_matrix = model.get_topics()
    with open(saved_file, "w") as writer:
        for i, word in dictionary.items():
            writer.write(f"{word} {' '.join([str(n) for n in topics_matrix[:, i]])}\n")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    extra_args = [
        {"flags": ["-dn", "--dataset_names"], "type": str, "target": None},
        # preprocess parameters
        {"flags": ["-pm", "--process_method"], "type": str, "target": None},
        {"flags": ["-dl", "--do_lemma"], "type": int, "target": None},
        {"flags": ["-ab", "--add_bi"], "type": int, "target": None},
        {"flags": ["-mc", "--min_count"], "type": int, "target": None},
        {"flags": ["-nob", "--no_below"], "type": int, "target": None},
        {"flags": ["-noa", "--no_above"], "type": float, "target": None},
        # topic parameters
        {"flags": ["-nt", "--num_topics"], "type": str, "target": None},
        {"flags": ["-cm", "--c_methods"], "type": str, "target": None},

    ]
    args, options = init_args(), customer_args(extra_args)
    config_parser = ConfigParser.from_args(args, options)
    config = config_parser.config
    do_lemma, add_bi = config.get("do_lemma", False), config.get("add_bi", False)
    dataset_names = config.get("dataset_names", "News26_MIND15")
    saved_path = Path(get_project_root()) / "saved" / "topic_embed"
    os.makedirs(saved_path)

    docs_token = []
    for dataset_name in dataset_names.split("_"):
        docs_token.extend(load_docs(dataset_name, config.get("process_method", "aggressive")))

    filter_dict = filter_tokens(docs_token, no_below=config.get("no_below", 20), no_above=config.get("no_above", 0.5))
    corpus_filter = get_bow_corpus(docs_token, filter_dict)
    for num_topic in config.get("num_topics", "10,50").split(","):
        lda = lda_model(filter_dict, corpus_filter, passes=1, num_topics=num_topic)
        save_topic_embed(lda, filter_dict, saved_path / f"topic_embed_{dataset_names}_{num_topic}.txt")
        for c_method in config.get("c_methods", "c_npmi,c_v").split(","):
            evaluate_topics(lda, corpus_filter, docs_token, filter_dict, c_method=c_method)
