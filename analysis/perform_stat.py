import os
import pandas as pd
import numpy as np

from pathlib import Path
from itertools import product
from news_recommendation.config import load_cmd_line
from news_recommendation.utils import get_project_root, del_index_column, write_to_file


def get_mean_std(values, r: int = 2):
    return f"{np.round(np.mean(values), r)}" + u"\u00B1" + f"{np.round(np.std(values), r)}"


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    root_path = Path(cmd_args.get("root_path", Path(get_project_root()) / "saved"))
    perform_dir = root_path / "performance"
    saved_path = root_path / "stat"
    os.makedirs(saved_path, exist_ok=True)
    latex_dir = saved_path / "latex"
    os.makedirs(latex_dir, exist_ok=True)
    input_path = cmd_args.get("input_path", None)
    if input_path is None:
        input_file = cmd_args.get("input_file", "MIND15-keep_all-base.csv")
        input_path = perform_dir / input_file
    input_file_name = input_path.name
    per_df = pd.read_csv(input_path)
    per_df = del_index_column(per_df.drop_duplicates())
    group_by = cmd_args.get("group_by", ["arch_type", "variant_name"])
    defaults = ["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"] if "RS" in input_file_name else ["accuracy", "macro_f"]
    metrics = cmd_args.get("metrics", defaults)  # For NC: accuracy,macro_f
    group_set = cmd_args.get("group_set", ["val"])  # For NC: val,test
    extra_stat = cmd_args.get("extra_stat", [])
    metrics_per = [f"{d}_{m}" for d, m in product(group_set, metrics)]
    columns = group_by + metrics_per + extra_stat
    stat_df = pd.DataFrame(columns=columns)
    for group_names, group in per_df.groupby(group_by):
        mean_values = [get_mean_std(group[m].values * 100) for m in metrics_per]
        extra_values = [get_mean_std(group[m].values, r=4) for m in extra_stat]
        mean_series = pd.Series(list(group_names) + mean_values + extra_values, index=columns)
        stat_df = stat_df.append(mean_series, ignore_index=True)
    output_path = cmd_args.get("output_path", None)
    if output_path is None:  # if not specified, save to default stat file
        output_file = cmd_args.get("output_file", input_file_name)
        output_path = saved_path / output_file
    if os.path.exists(output_path):
        old_stat_df = pd.read_csv(output_path)
        old_stat_df = old_stat_df.append(stat_df, ignore_index=True)
        stat_df = old_stat_df
    stat_df = del_index_column(stat_df).drop_duplicates()
    stat_df.to_csv(output_path)  # save to csv
    # save to file in latex format
    write_to_file(latex_dir / input_file_name.replace(".csv", ".txt"), stat_df.to_latex())
