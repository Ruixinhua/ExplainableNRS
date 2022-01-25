import logging
import os
import pandas as pd

from pathlib import Path
from experiment import init_args
from utils import load_dataset_df, tokenize, get_project_root, write_to_file, evaluate_entropy, del_index_column
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel


def lda_model(dictionary, corpus, **kwargs):
    temp = dictionary[0]  # This is only to "load" the dictionary.
    # Make an index to word dictionary.
    args_dict = {
        "corpus": corpus, "id2word": dictionary.id2token, "chunksize": kwargs.get("chunksize", 2000),
        "alpha": kwargs.get("alpha", "auto"), "eta": kwargs.get("eta", "auto"), "passes": kwargs.get("passes", 10),
        "iterations": kwargs.get("iterations", 400), "num_topics": kwargs.get("num_topics", 50),
        "eval_every": kwargs.get("eval_every", None)
    }
    return LdaModel(**args_dict)


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


def evaluate_topics(model, corpus, docs, dictionary, num_topics=50, method="c_npmi", topn=20, file=None):
    top_topics = model.top_topics(corpus, texts=docs, dictionary=dictionary, coherence=method, topn=topn)
    topics = [" ".join([f"{t[1]}: {t[0]}" for t in topic[0]]) for topic in top_topics]
    # Average topic coherence is the sum of topic coherence of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    topics.append(f'Average topic coherence({method}): %.4f. \n' % avg_topic_coherence)
    write_to_file(file, topics, mode="a+")
    return avg_topic_coherence


def load_docs(name, method):
    df, _ = load_dataset_df(name)
    docs = [tokenize(d, method) for d in df["data"].values]
    if args.do_lemma:
        docs = lemmatize(docs)
    if args.add_bi:
        docs = add_bigram(docs, args.min_count)
    return docs


def save_topic_embed(model, dictionary, saved_file):
    topics_matrix = model.get_topics()
    with open(saved_file, "w") as writer:
        for i, word in dictionary.items():
            writer.write(f"{word} {' '.join([str(n) for n in topics_matrix[:, i]])}\n")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    cus_args = [
        {"flags": ["-dn", "--dataset_names"], "type": str, "default": "News26"},
        # preprocess parameters
        {"flags": ["-pm", "--process_method"], "type": str, "default": "keep_all"},
        {"flags": ["-dl", "--do_lemma"], "type": int, "default": 0},
        {"flags": ["-ab", "--add_bi"], "type": int, "default": 0},
        {"flags": ["-mc", "--min_count"], "type": int, "default": 200},
        {"flags": ["-nob", "--no_below"], "type": int, "default": 20},
        {"flags": ["-noa", "--no_above"], "type": float, "default": 0.5},
        # topic parameters
        {"flags": ["-nt", "--num_topics"], "type": str, "default": "10,30,50,70,100,120,150,180,200"},
        {"flags": ["-cm", "--c_methods"], "type": str, "default": "c_npmi,c_v"},
        {"flags": ["-tn", "--topn"], "type": int, "default": 25},
        {"flags": ["-pa", "--passes"], "type": int, "default": 10},
    ]
    args = init_args(cus_args).parse_args()
    dataset_names = args.dataset_names
    saved_path = Path(get_project_root()) / "saved" / "topic_model"
    os.makedirs(saved_path, exist_ok=True)

    docs_token = []
    for dataset_name in dataset_names.split("_"):
        docs_token.extend(load_docs(dataset_name, args.process_method))

    filter_dict = filter_tokens(docs_token, no_below=args.no_below, no_above=args.no_above)
    corpus_filter = get_bow_corpus(docs_token, filter_dict)
    model_name = "LDA"
    stat_path = saved_path / "stat"
    os.makedirs(stat_path, exist_ok=True)
    stat_file = stat_path / f"{dataset_names}_{model_name}.csv"
    stat_df = pd.DataFrame() if not os.path.exists(stat_file) else pd.read_csv(stat_file)
    for num_topic in args.num_topics.split(","):
        saved_name = f"{dataset_names}_{num_topic}_{model_name}"
        topic_model = lda_model(filter_dict, corpus_filter, passes=args.passes, num_topics=int(num_topic))
        save_topic_embed(topic_model, filter_dict, saved_path / f"{saved_name}.txt")
        stat_dict = {
            "model": model_name, "num_topic": num_topic, "process_method": args.process_method, "topn": args.topn,
            "#Vocabulary": len(filter_dict)
        }
        for c_method in args.c_methods.split(","):
            log_path = saved_path / "log"
            os.makedirs(log_path, exist_ok=True)
            score = evaluate_topics(topic_model, corpus_filter, docs_token, filter_dict, method=c_method,
                                    topn=args.topn, file=log_path / f"{saved_name}.txt", num_topics=int(num_topic))
            stat_dict[c_method] = score
        token_entropy, topic_entropy = evaluate_entropy(topic_model.get_topics())
        stat_dict.update({"token_entropy": token_entropy, "topic_entropy": topic_entropy})
        stat_df = stat_df.append(pd.Series(stat_dict), ignore_index=True)
    del_index_column(stat_df).to_csv(stat_file)
