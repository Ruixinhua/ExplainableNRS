# -*- coding: utf-8 -*-
# @Organization  : UCD
# @Author        : Dairui Liu
# @Time          : 16/01/2023 21:43
# @Function      : Convert statistic data to .dat format
import os
import pandas as pd

from pathlib import Path
from config import load_cmd_line
from utils import get_project_root

if __name__ == "__main__":
    cmd_args = load_cmd_line()
    stat_path = Path(cmd_args.get("stat_path", None))
    r"""
    C:\Users\Rui\Documents\Explainable_AI\explainable_nrs\saved\stat\coherence_score\lda_percents_scores.csv
    
    """
    saved_dir = Path(cmd_args.get("saved_dir", Path(get_project_root()) / "saved" / "stat" / "dat_files"))
    os.makedirs(saved_dir, exist_ok=True)
    output_name = cmd_args.get("output_name", stat_path.name.split(".")[0])
    x_columns = cmd_args.get("x_columns", "topic_num".split(","))
    y_columns = cmd_args.get("y_columns", "Top-10%".split(","))
    split_columns = cmd_args.get("split_columns", "metric".split(","))
    if stat_path is None:
        raise ValueError("Please specify stat_path")
    stat_df = pd.read_csv(stat_path)
    dat_df = pd.DataFrame()
    for col, group_df in stat_df.groupby(split_columns):
        group_df = group_df.sort_values(by=x_columns)
        y_cols = []
        for y_col in y_columns:
            tmp_col = [f"{col}_{y_col}", f"{col}_{y_col}_error"]
            y_cols.extend(tmp_col)
            group_df[tmp_col] = group_df[y_col].str.split(u"\u00B1", expand=True)
        tmp_df = group_df[x_columns+y_cols]
        # merge data_df and tmp_df on x_columns
        dat_df = dat_df.merge(tmp_df, on=x_columns, how="outer") if not dat_df.empty else tmp_df
    dat_df.to_csv(saved_dir / f"{output_name}.dat", sep="\t", index=False)
