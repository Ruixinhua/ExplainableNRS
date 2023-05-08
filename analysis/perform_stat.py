import os
import pandas as pd
import numpy as np

from pathlib import Path
from itertools import product

from pandas.errors import InvalidIndexError

from modules.config import load_cmd_line
from modules.utils import get_project_root, del_index_column, write_to_file


def get_mean_std(values, r: int = 2):
    return f"{np.round(np.mean(values), r)}" + u"\u00B1" + f"{np.round(np.std(values), r)}"


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    root_path = Path(cmd_args.get("root_path", Path(get_project_root()) / "saved"))
    perform_dir = root_path / "performance"
    saved_path = root_path / "stat"
    latex_dir = saved_path / "latex"
    os.makedirs(latex_dir, exist_ok=True)
    if "input_path" not in cmd_args:
        input_file = cmd_args.get("input_file", None)
        if input_file is None:
            raise ValueError("Please specify input_path or input_file")
        if os.path.exists(input_file):
            input_path = Path(input_file)
        else:
            input_path = perform_dir / input_file
    else:
        input_path = Path(cmd_args["input_path"])
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found")

    input_file_name = input_path.name
    per_df = pd.read_csv(input_path)
    per_df = del_index_column(per_df.drop_duplicates())
    defaults = ["subset_type", "arch_type", "variant_name"] if "RS" in input_file_name else ["arch_type", "variant_name"]
    group_by = cmd_args.get("group_by", defaults)
    defaults = ["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"] if "RS" in input_file_name else ["accuracy", "macro_f"]
    metrics = cmd_args.get("metrics", defaults)  # For NC: accuracy,macro_f
    group_set = cmd_args.get("group_set", ["val"] if "RS" in input_file_name else ["val", "test"])  # For NC: val,test
    extra_stat = cmd_args.get("extra_stat", ["Total Time"])
    metrics_per = [f"{d}_{m}" for d, m in product(group_set, metrics)]
    columns = group_by + metrics_per + extra_stat
    stat_df = pd.DataFrame(columns=columns)
    for group_names, group in per_df.groupby(group_by):
        mean_values = [get_mean_std(group[m].values * 100) if group[m].values[0] < 1 else get_mean_std(group[m].values)
                       for m in metrics_per]
        extra_values = [get_mean_std(group[m].values, r=4) for m in extra_stat]
        mean_series = pd.Series(list(group_names) + mean_values + extra_values, index=columns)
        stat_df = stat_df.append(mean_series, ignore_index=True)
    if "output_path" not in cmd_args:  # if not specified, save to default stat file
        output_path = saved_path
    else:
        output_path = Path(cmd_args["output_path"])
    os.makedirs(output_path, exist_ok=True)
    output_file = cmd_args.get("output_file", input_file_name)
    output_path = output_path / output_file
    if os.path.exists(output_path):
        old_stat_df = pd.read_csv(output_path)
        try:  # if the old file has extra columns, delete them
            old_stat_df = old_stat_df.drop(columns=old_stat_df.columns.difference(stat_df.columns))
            old_stat_df = old_stat_df.append(stat_df, ignore_index=True)
            stat_df = old_stat_df
        except InvalidIndexError:
            pass
    stat_df = del_index_column(stat_df).drop_duplicates()
    stat_df.to_csv(output_path)  # save to csv
    print("write statistic data to file:", output_path)
    # save to file in latex format
    write_to_file(latex_dir / input_file_name.replace(".csv", ".txt"), stat_df.to_latex())
    print("save latex data to file:", latex_dir / input_file_name.replace(".csv", ".txt"))
