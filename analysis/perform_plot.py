# -*- coding: utf-8 -*-
# @Organization  : UCD
# @Author        : Dairui Liu
# @Time          : 12/01/2023 03:30
# @Function      : Plot NR performance


import os

import pandas as pd
import copy
import plotly.graph_objects as go
from modules.config import load_cmd_line
from pathlib import Path

from utils import get_project_root


def take_mean(values):
    return float(values.split(u"\u00B1")[0])


def plot(stat_df, filename, title=None, yaxis_title=None):
    plot_df = copy.deepcopy(stat_df[["head_num"] + metrics])
    fig = go.Figure()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    markers = ["circle", "square", "diamond", "x", "star"]
    for name in metrics:
        plot_df[name] = plot_df[name].apply(take_mean)
    plot_df["topic_num"] = plot_df.head_num.apply(lambda h: str(h))
    for n, color, marker, mn in zip(metrics, colors, markers, metrics_name):
        kwargs = dict(marker_symbol=marker, marker_size=8, line=dict(color=color, width=2))
        fig.add_trace(go.Scatter(x=plot_df["topic_num2"], y=plot_df[n], name=mn, **kwargs))
    fig.update_layout(title=title, xaxis_title='#Topic', template="plotly_white", width=1000, height=500,
                      yaxis_title=yaxis_title)
    fig.show()
    fig.write_image(image_saved_dir / f"{filename}.png")


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    rs_stat_path = cmd_args.get("rs_stat_path", None)
    output_name = cmd_args.get("output_name", "BATM_ATT")
    plot_set = cmd_args.get("plot_set", "test")
    image_saved_dir = Path(cmd_args.get("image_saved_dir", f"{get_project_root()}/saved/plot_images"))
    os.makedirs(image_saved_dir, exist_ok=True)
    perform_df = pd.DataFrame()
    metrics = [f"{plot_set}_{m}" for m in ["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"]]
    metrics_name = ["AUC", "MRR", "nDCG@5", "nDCG@10"]
    yaxis_temp = "Average Scores"
    if rs_stat_path is None:
        raise ValueError("Please specify rs_stat_path")
    rs_df = pd.read_csv(rs_stat_path)
    rs_title = "Mean Recommendation Performance Scores of Different Topic Numbers Setting"
    plot(rs_df, "w2v_sim", output_name, yaxis_temp)
