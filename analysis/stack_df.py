# -*- coding: utf-8 -*-
# @Organization  : UCD
# @Author        : Dairui Liu
# @Time          : 06/05/2023 17:15
# @Function      : Stack the performance results of different runs
import pandas as pd
from pathlib import Path
from modules.config import load_cmd_line


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    input_dir = cmd_args.get("input_dir", None)
    output_path = cmd_args.get("output_path", None)
    if input_dir is None or output_path is None:
        raise ValueError("Please specify input_dir")
    input_dir = Path(input_dir)
    stack_df = pd.DataFrame()
    for file in input_dir.iterdir():
        if file.is_file() and file.suffix == ".csv":
            per_df = pd.read_csv(file)
            per_df = per_df.drop_duplicates()
            stack_df = stack_df.append(per_df, ignore_index=True)
    stack_df.to_csv(output_path, index=False)
