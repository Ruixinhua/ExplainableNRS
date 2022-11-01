# -*- coding: utf-8 -*-
# @Organization  : UCD
# @Author        : Dairui Liu
# @Time          : 28/10/2022 15:02
# @Function      : Plot performance evaluation results and topic coherence evaluation results


import os

import pandas as pd
import copy
import plotly.graph_objects as go
from modules.config import load_cmd_line
from pathlib import Path

from utils import get_project_root


def take_mean(values):
    return float(values.split(u"\u00B1")[0])


def coherence_plot(stat_df, suffix, title=None, yaxis_title=None, task="NC"):
    plot_names = [f"PP{i}_{suffix}" for i in pps] + [f"original_{suffix}"]
    plot_df = copy.deepcopy(stat_df[["head_num"] + plot_names])
    fig = go.Figure()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    markers = ["circle", "square", "diamond", "x", "star"]
    names = [f"Post-Process-{n}" for n in pps] + ["Without PP"]
    filename = f"{task}_{suffix}"
    for name in plot_names:
        plot_df[name] = plot_df[name].apply(take_mean)
    plot_df["head_num"] = plot_df.head_num.apply(lambda h: str(h))
    for n, color, marker, mn in zip(plot_names, colors, markers, names):
        kwargs = dict(marker_symbol=marker, marker_size=8, line=dict(color=color, width=2))
        fig.add_trace(go.Scatter(x=plot_df["head_num"], y=plot_df[n], name=mn, **kwargs))
    if lda_stat_path is not None and add_lda:
        kwargs = dict(marker_symbol="triangle-up", marker_size=8, line=dict(color="#19D3F3", width=2))
        fig.add_trace(go.Scatter(x=plot_df["head_num"], y=lda_df[suffix], name="LDA", **kwargs))
        filename += "_lda"
    fig.update_layout(title=title, xaxis_title='#Topic', template="plotly_white", width=1000, height=500,
                      yaxis_title=yaxis_title)
    fig.show()
    fig.write_image(image_saved_dir / f"{filename}.png")


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    nc_stat_path = cmd_args.get("nc_stat_path", None)
    rs_stat_path = cmd_args.get("rs_stat_path", None)
    lda_stat_path = cmd_args.get("lda_stat_path", None)
    add_lda = cmd_args.get("add_lda", False)
    image_saved_dir = Path(cmd_args.get("image_saved_dir", f"{get_project_root()}/saved/plot_images"))
    os.makedirs(image_saved_dir, exist_ok=True)
    perform_df = pd.DataFrame()
    pps = [100, 60, 30, 10]
    yaxis_temp = "Average TC"
    if lda_stat_path is not None and add_lda:
        lda_df = pd.read_csv(lda_stat_path)
        lda_df["w2v_sim"] = lda_df.W2v.apply(take_mean)
        lda_df["c_npmi"] = lda_df.NPMI.apply(take_mean)
    if nc_stat_path is None and rs_stat_path is None:
        raise ValueError("Please specify nc_stat_path or rs_stat_path")
    if nc_stat_path is not None:
        nc_df = pd.read_csv(nc_stat_path)
        nc_title = "Mean TC scores for BATM model with different post processing methods for NC"
        coherence_plot(nc_df, "w2v_sim", nc_title.replace("TC", "W2VSim"), yaxis_temp.replace("TC", "W2VSim"), "NC")
        coherence_plot(nc_df, "c_npmi", nc_title.replace("TC", "NPMI"), yaxis_temp.replace("TC", "NPMI"), "NC")
        stat_col = ["head_num", "test_accuracy", "test_macro_f"]
        perform_df = perform_df.append(nc_df[stat_col], ignore_index=True)
    if rs_stat_path is not None:
        rs_df = pd.read_csv(rs_stat_path)
        rs_title = "Mean TC scores for BATM model with different post processing methods for RS"
        coherence_plot(rs_df, "w2v_sim", rs_title.replace("TC", "W2VSim"), yaxis_temp.replace("TC", "W2VSim"), "RS")
        coherence_plot(rs_df, "c_npmi", rs_title.replace("TC", "NPMI"), yaxis_temp.replace("TC", "NPMI"), "RS")
