import os
import pandas as pd
import numpy as np

from pathlib import Path
from itertools import product
from news_recommendation.utils import get_project_root, del_index_column, write_to_file


def get_mean_std(values, r: int = 2):
    return f"{np.round(np.mean(values), r)}" + u"\u00B1" + f"{np.round(np.std(values), r)}"


if __name__ == "__main__":
    root_path = Path(get_project_root()) / "saved"
    perform_dir = root_path / "performance"
    saved_path = root_path / "stat"
    os.makedirs(saved_path, exist_ok=True)
    latex_dir = saved_path / "latex"
    os.makedirs(latex_dir, exist_ok=True)
    input_file = perform_dir / "RS_BATM-MIND15-keep_all-None-evaluate_topic.csv"
    per_df = del_index_column(pd.read_csv(input_file).drop_duplicates())
    group_by = ["arch_type", "mind_type", "variant_name"]
    # test_args = ["head_num", "embedding_type", "base", "variant_name"]
    metrics = ["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"]
    metrics_per = [f"{d}_{m}" for d, m in product(["val"], metrics)]
    stat_df = pd.DataFrame(columns=group_by + metrics_per)
    for group_names, group in per_df.groupby(group_by):
        mean_values = [get_mean_std(group[m].values * 100) for m in metrics_per]
        mean_series = pd.Series(list(group_names)+mean_values, index=group_by + metrics_per)
        stat_df = stat_df.append(mean_series, ignore_index=True)
    stat_df.to_csv(saved_path / "RS-MIND15-BATM-small.csv")
