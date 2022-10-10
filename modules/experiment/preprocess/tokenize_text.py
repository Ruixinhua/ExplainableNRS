import os
import pandas as pd
import json
from modules.config import load_cmd_line
from modules.utils import read_json, word_tokenize, clean_df, get_project_root
from pathlib import Path


def load_mind_df(mind_types, mind_path=None):
    if mind_path is None:
        mind_path = get_project_root() / "dataset" / "MIND"
    set_types = ["train", "valid", "test"]
    news_df = pd.DataFrame()
    columns = ["news_id", "category", "subvert", "title", "abstract", "url", "entity", "ab_entity"]
    for mt in mind_types:
        for st in set_types:
            news_path = mind_path / mt / st / "news.tsv"
            article_path = mind_path / mt / st / "msn.json"
            if not os.path.exists(article_path):
                continue
            articles = read_json(article_path)
            df = pd.read_table(news_path, header=None, names=columns)
            df["body"] = df.news_id.apply(lambda nid: " ".join(articles[nid]) if nid in articles else "")
            news_df = news_df.append(df)
    news_df.drop_duplicates(inplace=True)
    return news_df


def save_jsonl(doc_df, path):
    with open(path, "w") as outfile:
        for line_no, (doc, doc_idx) in enumerate(zip(doc_df.docs, doc_df.news_id)):
            json_doc = json.dumps({"id": doc_idx, "text": doc})
            if line_no == 0:
                outfile.write(json_doc)
            else:
                outfile.write("\n" + json_doc)


def tokenize_mind(mind_type):
    df = clean_df(load_mind_df([mind_type], mind_path=Path(dataset_path) / "MIND").drop_duplicates())
    df["tokenized_text"] = df.title + " " + df.abstract + " " + df.body
    df["tokenized_text"] = df.tokenized_text.apply(lambda ws: " ".join([s for s in word_tokenize(ws) if s in ori_dict]))
    return df


def count_tokens(df):
    count = 0
    for texts in df.tokenized_text.values:
        count += len(texts.split())
    return count


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    mind_type = cmd_args.get("mind_type", "large")
    dataset_path = cmd_args.get("dataset_path", Path(get_project_root()) / "dataset")
    data_path = Path(dataset_path) / "data"
    ori_dict = read_json(Path(dataset_path) / "utils/word_dict/MIND_41059.json")
    mind_large_df = tokenize_mind(mind_type)
    mind_large_df["docs"] = mind_large_df.tokenized_text
    save_jsonl(mind_large_df, data_path / f"raw/MIND_{mind_type}_original.jsonl")
    print("save tokenized json to", data_path / f"raw/MIND_{mind_type}_original.jsonl")
