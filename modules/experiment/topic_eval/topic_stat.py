import pandas as pd
import os
import shutil

from analysis.perform_stat import get_mean_std
from config import load_cmd_line
from modules.utils import read_json, write_json
from pathlib import Path


def sum_between(df, left, right):
    return round(sum(df["NPMI"].between(left, right)) / len(df), 4)


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    stat_topic_dir = Path(cmd_args.get("stat_topic_dir", "saved/stat/topics"))
    topic_saved_dir = Path(cmd_args.get("topic_saved_dir", "saved/models/MIND15/keep"))
    interval_columns = ["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-1"]
    for d in os.listdir(topic_saved_dir):
        config_dict = read_json(Path(topic_saved_dir, str(d), "config.json"))
        topic_stats = []
        word_dict_size = config_dict["word_dict_file"].split(".")[0].split("_")[-1]
        new_topic_dir = stat_topic_dir / word_dict_size / config_dict["arch_type"]
        os.makedirs(new_topic_dir, exist_ok=True)
        write_json(config_dict, new_topic_dir / "config.json")
        for td in os.listdir(Path(topic_saved_dir, str(d))):
            topic_dir = Path(topic_saved_dir, str(d), str(td))
            if not topic_dir.is_dir() or not topic_dir.name.startswith("topics"):
                continue
            topic_file = [str(f) for f in os.listdir(topic_dir) if "c_npmi" in str(f)]
            topic_path = topic_dir / topic_file[0]
            topic_list = {"NPMI": [], "terms": []}
            with open(topic_path, encoding="utf-8") as rd:
                for line in rd:
                    npmi, terms = line.split(":")
                    if npmi.startswith("Average"):
                        break
                    topic_list["NPMI"].append(float(npmi))
                    topic_list["terms"].append(terms.strip("\n"))
            topic_df = pd.DataFrame.from_dict(topic_list)
            seed = str(td).split("_")[1]
            topic_num = len(topic_df)
            topic_stats.append([topic_num, seed, sum_between(topic_df, 0, 0.1), sum_between(topic_df, 0.1, 0.2),
                                sum_between(topic_df, 0.2, 0.3), sum_between(topic_df, 0.3, 1)])
            c_npmi = topic_path.name.replace('.txt', '').replace('topic_list_', '')
            new_filename = f"{config_dict['topic_variant']}_{c_npmi}_{topic_num}_{seed}.txt"
            shutil.copy(topic_path, new_topic_dir / new_filename)
        topic_stat_df = pd.DataFrame.from_records(topic_stats, columns=["topic_number", "seed"] + interval_columns)
        topic_stat_df.to_csv(new_topic_dir / "topics_stat.csv")
        topic_mean_stat = []
        for topic_num, group in topic_stat_df.groupby("topic_number"):
            mean_values = [get_mean_std(group[m].values * 100) for m in interval_columns]
            topic_mean_stat.append([topic_num]+mean_values)
        mean_stat_df = pd.DataFrame.from_records(topic_mean_stat, columns=["topic_number"]+interval_columns)
        mean_stat_df.to_csv(new_topic_dir / "topics_mean_stat.csv")
