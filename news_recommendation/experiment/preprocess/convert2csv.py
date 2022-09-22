from pathlib import Path

import pandas as pd
import json
from config import load_cmd_line
from utils import read_json, write_json, get_project_root


def save_new_voc(path):
    vocab = read_json(path)
    vocab_new = {"[UNK]": 0}
    vocab_new.update({w: i+1 for w, i in vocab.items()})
    print(len(vocab_new))
    write_json(vocab_new, Path(dataset_path) / f"utils/word_dict/MIND_{mind_type}_{len(vocab_new)}.json")
    return vocab_new


def save_tokenized_text(path):
    processed_mind_dict = {"news_id": [], "tokenized_text": []}
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            data = json.loads(line)
            processed_mind_dict["news_id"].append(data["id"])
            processed_mind_dict["tokenized_text"].append(data["tokenized_text"])
    mind_processed_df = pd.DataFrame.from_dict(processed_mind_dict, orient="columns")
    mind_processed_df.to_csv(Path(dataset_path) / f"data/MIND_{mind_type}_{len(vocab)}.csv")
    return mind_processed_df


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    mind_type = cmd_args.get("mind_type", "large")
    dataset_path = cmd_args.get("dataset_path", Path(get_project_root()) / "dataset")
    processed_path = cmd_args.get("processed_path", None)
    data_path = Path(dataset_path) / "data"
    vocab = save_new_voc(Path(processed_path) / "vocab.json")
    save_tokenized_text(Path(processed_path) / "train.metadata.jsonl")
