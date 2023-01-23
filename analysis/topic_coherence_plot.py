# -*- coding: utf-8 -*-
# @Organization  : UCD
# @Author        : Dairui Liu
# @Time          : 20/01/2023 04:25
# @Function      : plot topic coherence boxplot and lineplot of the LDA model and our models
import os
import copy
from collections import defaultdict

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

from analysis import get_mean_std, take_mean
from modules.config import load_cmd_line
from modules.utils import get_project_root

colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
markers = ["circle-open", "square-open", "diamond-open", "x-open", "star-open", "triangle-up-open"]
# dashstyle = ["solid", "dot", "dash", "dashdot", "solid"]
dashstyle = ["solid"] * 10
marker_size = 10
margin = 10
ranges = {"npmi": [0.2, 0.5], "w2v": [0.35, 0.6]}
percentages = [10]  # the percentage of top topics


def read_scores(reader, topic_num):
    return sorted([eval(next(reader)) for _ in range(int(topic_num))], reverse=True)


def load_lda_scores(score_dir):
    scores = []
    for file in os.scandir(score_dir):
        method, topic_num = file.name.split("_")[0], file.name.split("_")[2]
        with open(file) as reader:
            scores.append([method, topic_num, read_scores(reader, topic_num)])
    return pd.DataFrame.from_records(scores, columns=["metric", "topic_num", "scores"])


def cal_per_scores(scores, p):
    num = int(p * len(scores) / 100)
    return round(sum(scores[:num]) / num, 4)


def cal_group_scores(df, group_keys, percents=None):
    if percents is None:
        percents = [10, 30, 50, 100]
    group_scores = []
    for keys, group in df.groupby(group_keys):
        scores = [get_mean_std([cal_per_scores(s, p) for s in group["scores"].values], 4) for p in percents]
        group_scores.append(list(keys) + scores)
    return pd.DataFrame.from_records(group_scores, columns=group_keys + [f"Top-{p}%" for p in percents])


def load_batm_scores(score_dir):
    score = []
    for d in os.scandir(score_dir):
        if not d.is_dir():
            continue
        topic_num = d.name.split("_")[2]
        for file in os.scandir(d):
            name = file.name.split("_")
            pp_method = name[0]
            metric = name[1] if name[1] == "w2v" else "npmi"
            score.append([metric, pp_method, topic_num, read_scores(open(file), topic_num)])
    return pd.DataFrame.from_records(score, columns=["metric", "pp_method", "topic_num", "scores"])


def sort_df(df, sort_key):
    df[sort_key] = df[sort_key].astype(int)  # sort by int
    df = df.sort_values(by=sort_key)
    df[sort_key] = df[sort_key].astype(str)  # convert back to str
    return df


def coherence_percents_plot(stat_df, lda_df, image_saved_dir, percents=None):
    if percents is None:
        percents = [10, 30, 50, 100]
    columns = [f"Top-{p}%" for p in percents]
    plot_rows = [f"PP{i}" for i in sorted(pps, reverse=True)] + ["original"]  # from upper to lower
    names = [f"PP-{n}" for n in sorted(pps, reverse=True)] + ["Without PP"]
    plot_df = copy.deepcopy(stat_df)
    lda_plot_df = copy.deepcopy(lda_df)
    for name in columns:
        plot_df[name] = plot_df[name].apply(take_mean)
        lda_plot_df[name] = lda_plot_df[name].apply(take_mean)
    for metric, group in plot_df.groupby(["metric"]):
        for topn in columns:
            fig = go.Figure()
            row = lda_plot_df[lda_plot_df["metric"] == metric]
            kwargs = dict(marker_symbol=markers[-1], marker_size=marker_size,
                          line=dict(color="#19D3F3", width=2, dash=dashstyle[0]))
            fig.add_trace(go.Scatter(x=row["topic_num"], y=row[topn], name="LDA", **kwargs))
            tasks = group["task"].unique()
            for i in range(len(tasks)):
                for j in range(len(plot_rows)):
                    index = i * len(plot_rows) + j
                    row = group[(group["task"] == tasks[i]) & (group["pp_method"] == plot_rows[j])]
                    kwargs = dict(marker_symbol=markers[index], marker_size=marker_size,
                                  line=dict(color=colors[index], width=2, dash=dashstyle[index]))
                    fig.add_trace(go.Scatter(x=row["topic_num"], y=row[topn], name=f"{tasks[i]}-{names[j]}", **kwargs))
            # title = f"{topn} {metric} Coherence Scores for {task} task"
            fig.update_layout(xaxis_title='#Topic', template="plotly_white", width=600, height=400,
                              yaxis_title=f"Mean {metric.upper()}", margin=dict(l=margin, r=margin, t=margin, b=margin))
            fig.update_yaxes(gridcolor='grey', griddash="dash", showline=True, linewidth=2, gridwidth=1,
                             mirror=True, linecolor='black')
            fig.update_xaxes(showgrid=False, showline=True, linewidth=2, mirror=True, linecolor='black')
            fig.update_yaxes(range=ranges[metric])
            os.makedirs(image_saved_dir, exist_ok=True)
            fig.write_image(image_saved_dir / f"{metric}_{topn}.png")
    print(f"Coherence scores plot saved in {image_saved_dir}")


def plot_box(topic_scores_df, lda_scores_df, image_saved_dir):
    plot_df = copy.deepcopy(topic_scores_df)
    lda_df = copy.deepcopy(lda_scores_df)
    plot_rows = ["original"] + [f"PP{i}" for i in pps]  # from left to right
    x_names = {f"PP{i}": f"PP-{i}" for i in pps}
    x_names["original"] = "Without PP"
    topic_scores_all, lda_scores_all = defaultdict(lambda: defaultdict(lambda: [])), defaultdict(lambda: [])
    plot_df = plot_df.drop_duplicates(subset=["metric", "pp_method", "topic_num", "task"])
    lda_df = lda_df.drop_duplicates(subset=["metric", "topic_num"])
    box_kwargs = dict(jitter=0.4, whiskerwidth=0.2, marker=dict(size=1, color='rgb(0, 0, 0)'),
                      line=dict(width=1), boxpoints="all")
    for (metric, topic_num), group in plot_df.groupby(["metric", "topic_num"]):
        fig = go.Figure()
        tasks = group["task"].unique()
        for i in range(len(tasks)):
            for j in range(len(plot_rows)):
                index = i * len(plot_rows) + j
                score = np.stack(group[group["pp_method"] == plot_rows[j]]["scores"].values).flatten()
                box_kwargs["fillcolor"] = colors[index]
                fig.add_trace(go.Box(y=score, name=f"{tasks[i]}-{x_names[plot_rows[j]]}", **box_kwargs))
                topic_scores_all[metric][f"{tasks[i]}-{plot_rows[j]}"].extend(score)
        lda_rows = lda_df[(lda_df["metric"] == metric) & (lda_df["topic_num"] == topic_num)]
        lda_score = np.stack(lda_rows["scores"].values).flatten()
        lda_scores_all[metric].extend(lda_score)
        box_kwargs["fillcolor"] = colors[-1]
        fig.add_trace(go.Box(y=lda_score, name="LDA", **box_kwargs))
        # title = f"Box plot of {metric} scores for {topic_num} topics"
        fig.update_layout(template="plotly_white", width=1000, height=200, yaxis_title=f"{metric.upper()} score",
                          margin=dict(l=10, r=10, t=10, b=10))
        os.makedirs(image_saved_dir, exist_ok=True)
        fig.write_image(image_saved_dir / f"{metric}_{topic_num}.png")
    for metric, scores in topic_scores_all.items():
        fig = go.Figure()
        for ((name, score), color) in zip(scores.items(), colors):
            box_kwargs["fillcolor"] = color
            task, n = name.split("-")
            fig.add_trace(go.Box(y=score, name=f"{task}-{x_names[n]}", **box_kwargs))
        box_kwargs["fillcolor"] = colors[-1]
        fig.add_trace(go.Box(y=lda_scores_all[metric], name="LDA", **box_kwargs))
        # title = f"Box plot of {metric} scores for all topics"
        fig.update_layout(template="plotly_white", width=1000, height=200, yaxis_title=f"{metric.upper()} Score",
                          margin=dict(l=20, r=20, t=20, b=20))
        fig.write_image(image_saved_dir / f"{metric}_all.png")
        print("save box plots to", image_saved_dir)


def get_scores_percents_df(coherence_dir):
    coherence_dir = Path(coherence_dir)
    score = sort_df(load_batm_scores(coherence_dir), "topic_num")
    percents_df = cal_group_scores(score, ["pp_method", "metric", "topic_num"], percentages)
    percents_df = sort_df(percents_df, "topic_num")  # sort by topic number
    saved_path = saved_dir / f"{saved_name}.csv"
    percents_df.to_csv(saved_path, index=False)
    print("save the batm coherence scores on path:", saved_path)
    return score, percents_df


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    nc_coherence_dir = cmd_args.get("nc_coherence_dir", None)  # the directory of topic list
    nr_coherence_dir = cmd_args.get("nr_coherence_dir", None)  # the directory of topic list
    lda_coherence_dir = cmd_args.get("lda_coherence_dir", None)  # the directory of lda terms
    saved_name = cmd_args.get("saved_name", None)  # the name of the saved coherence scores
    if nc_coherence_dir is None or lda_coherence_dir is None:
        raise ValueError("The directory of batm model and lda coherence scores should be provided.")
    saved_dir = Path(cmd_args.get("save_dir", f"{get_project_root()}/saved/stat/coherence_score"))
    plot_dir = Path(cmd_args.get("plot_dir", f"{get_project_root()}/saved/plot_images/percents_coherence/NC_NR"))
    box_dir = Path(cmd_args.get("box_dir", f"{get_project_root()}/saved/plot_images/coherence_box_plot/NC_NR"))
    coherence_table_dir = Path(cmd_args.get("coherence_table_dir", f"{get_project_root()}/saved/stat/percents_table"))
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(coherence_table_dir, exist_ok=True)
    pps = [60]  # the number of post-processing
    lda_scores = sort_df(load_lda_scores(lda_coherence_dir), "topic_num")
    lda_percents_scores_df = cal_group_scores(lda_scores, ["metric", "topic_num"], percentages)
    lda_percents_scores_df = sort_df(lda_percents_scores_df, "topic_num")  # sort by topic number
    lda_percents_scores_df.to_csv(saved_dir / "lda_percents_scores.csv", index=False)
    print("save the lda coherence scores on path:", saved_dir / "lda_percents_scores.csv")
    nc_coherence_dir = Path(nc_coherence_dir)
    # task = nc_coherence_dir.name.split("_")[0]
    nc_scores, nc_percents_df = get_scores_percents_df(nc_coherence_dir)
    nr_scores, nr_percents_df = get_scores_percents_df(nr_coherence_dir)
    nc_percents_df["task"] = "NC"
    nr_percents_df["task"] = "NR"
    percents_all_df = pd.concat([nc_percents_df, nr_percents_df], axis=0)
    coherence_percents_plot(percents_all_df, lda_percents_scores_df, plot_dir, percents=percentages)
    # final_plot(percents_df, lda_percents_scores_df, plot_dir, percents=[100])
    nc_scores["task"] = "NC"
    nr_scores["task"] = "NR"
    scores_all = pd.concat([nc_scores, nr_scores], axis=0)
    plot_box(scores_all, lda_scores, box_dir)
    # coherence_table(percents_df, lda_percents_scores_df, coherence_table_dir, percents=percentages)
