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
    test_args = ["head_num", "embedding_type", "base", "variant_name"]
    for file in os.listdir(perform_dir):
        df_file = root_path / "performance" / file
        if os.path.isdir(df_file):
            continue
        per_df = del_index_column(pd.read_csv(df_file).drop_duplicates())
        stat_df = pd.DataFrame()
        args_list = file.replace(".csv", "").split("-")
        if "embedding_type" in args_list:
            group_by = ["arch_type", args_list[2]]
        else:
            group_by = ["arch_type", args_list[2], "variant_name", "max_length"]
        metrics_per = [f"{d}_{m}" for d, m in product(["val", "test"], ["accuracy", "macro_f"])]
        metrics_float = ["val_loss", "test_loss"]
        if "evaluate_topic" in args_list:
            metrics_float.extend(["NPMI", "CV", "token_entropy", "val_doc_entropy", "test_doc_entropy"])
        metrics = metrics_per + metrics_float
        for _, group in per_df.groupby(group_by):
            mean_values = [get_mean_std(group[m].values * 100) for m in metrics_per]
            mean_values.extend(get_mean_std(group[m].values, 4) for m in metrics_float)
            group = group.drop(columns=metrics + ["seed", "run_id", "dropout_rate"]).drop_duplicates()
            group[metrics] = pd.DataFrame([mean_values], index=group.index)
            stat_df = stat_df.append(group, ignore_index=True)
        stat_df.to_csv(saved_path / file)
        latex_columns = [args_list[2]] + ["val_accuracy", "val_macro_f"]
        if "evaluate_topic" in args_list:
            latex_columns.extend(["NPMI", "CV", ])
        for va, group in stat_df.groupby("variant_name"):
            os.makedirs(latex_dir / str(va), exist_ok=True)
            write_to_file(latex_dir / str(va) / file.replace(".csv", ".txt"), del_index_column(group[latex_columns].drop_duplicates()).to_latex())
