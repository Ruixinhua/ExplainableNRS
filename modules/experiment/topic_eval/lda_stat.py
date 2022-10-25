import pandas as pd
import os

from analysis.perform_stat import get_mean_std
from config import load_cmd_line
from modules.utils import read_json
from pathlib import Path


def sum_between(df, left, right):
    return round(sum(df["NPMI"].between(left, right)) / len(df), 4)


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    stat_topic_dir = Path(cmd_args.get("stat_topic_dir", "saved/stat/topics"))
    lda_dir = Path(cmd_args.get("lda_dir", "topics/mallet-MIND-large-original"))
    name = lda_dir.name.split("-")[-1]
    lda_stats = []
    topic_nums = [10, 50, 100, 200]
    for seed in os.listdir(lda_dir):
        if Path(lda_dir, str(seed)).is_dir():
            for num in topic_nums:
                metrics = read_json(Path(lda_dir, str(seed), f"metrics_{num}.json"))
                lda_stats.append([num, seed, metrics["npmi_mean"]])
    topic_stat_df = pd.DataFrame.from_records(lda_stats, columns=["topic_number", "seed", "NPMI"])
    topic_stat_df.to_csv(stat_topic_dir / f"lda_{name}.csv")
    topic_mean_stat = []
    for topic_num, group in topic_stat_df.groupby("topic_number"):
        topic_mean_stat.append([topic_num, get_mean_std(group["NPMI"].values, r=4)])
    mean_stat_df = pd.DataFrame.from_records(topic_mean_stat, columns=["topic_number", "NPMI"])
    mean_stat_df.to_csv(stat_topic_dir / f"lda_{name}_mean.csv")
