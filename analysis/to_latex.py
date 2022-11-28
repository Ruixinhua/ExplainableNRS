import os
import pandas as pd
import numpy as np

from pathlib import Path
from modules.config import load_cmd_line
from modules.utils import get_project_root, del_index_column, write_to_file


def get_mean_std(values, r: int = 2):
    return f"{np.round(np.mean(values), r)}" + u"\u00B1" + f"{np.round(np.std(values), r)}"


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    root_path = Path(cmd_args.get("root_path", Path(get_project_root()) / "saved"))
    stat_path = cmd_args.get("stat_path", root_path / "stat/performance_summary")
    stat_file = cmd_args.get("stat_file", "RS_Baselines.csv")
    latex_dir = cmd_args.get("latex_dir", root_path / "stat/latex/performance")
    os.makedirs(latex_dir, exist_ok=True)
    stat_df = pd.read_csv(stat_path / stat_file)
    stat_df = del_index_column(stat_df.drop_duplicates())
    write_to_file(Path(latex_dir, stat_file.replace(".csv", ".txt")), stat_df.to_latex(index=False))
