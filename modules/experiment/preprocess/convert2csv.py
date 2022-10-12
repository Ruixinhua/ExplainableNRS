import os
from pathlib import Path

import pandas as pd
import json
from config import load_cmd_line
from utils import read_json, write_json, get_project_root


def save_tokenized_text(word_dict_path, tokens_path):
    processed_mind_dict = {"news_id": [], "tokenized_text": []}
    vocab_old = read_json(word_dict_path)
    vocab_new = {"[UNK]": 0}
    with open(tokens_path, "r", encoding="utf-8") as infile:
        for line in infile:
            data = json.loads(line)
            processed_mind_dict["news_id"].append(data["id"])
            processed_mind_dict["tokenized_text"].append(data["tokenized_text"])
            for token in data["tokenized_text"].split():
                if token in vocab_old and token not in vocab_new:
                    vocab_new[token] = vocab_old[token]
    mind_processed_df = pd.DataFrame.from_dict(processed_mind_dict, orient="columns")
    mind_processed_df.to_csv(Path(dataset_path) / f"data/MIND_{mind_type}_{len(vocab_new)}.csv")
    if "[UNK]" not in vocab_old:
        vocab_new = {"[UNK]": 0}
        vocab_new.update({w: i+1 for w, i in vocab_old.items()})
    print(len(vocab_new))
    saved_dir = Path(dataset_path) / "utils/word_dict/post_process"
    os.makedirs(saved_dir, exist_ok=True)
    write_json(vocab_new, saved_dir / f"PP{Path(processed_path).name.split('-')[-1]}.json")
    return mind_processed_df


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    mind_type = cmd_args.get("mind_type", "large")
    dataset_path = cmd_args.get("dataset_path", Path(get_project_root()) / "dataset")
    processed_path = cmd_args.get("processed_path")
    vocab_dict_path = cmd_args.get("vocab_dict_path", Path(processed_path) / "vocab.json")
    data_path = Path(dataset_path) / "data"
    save_tokenized_text(vocab_dict_path, Path(processed_path) / "train.metadata.jsonl")
