import os
import copy
from collections import defaultdict

import matplotlib
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

from analysis import get_mean_std, take_mean
from modules.config import load_cmd_line
from modules.utils import get_project_root

colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
markers = ["circle", "square", "diamond", "x", "star"]
dashstyle = ["solid", "dot", "dash", "dashdot", "solid"]


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
    scores = []
    for d in os.scandir(score_dir):
        if not d.is_dir():
            continue
        topic_num = d.name.split("_")[-1]
        for file in os.scandir(d):
            name = file.name.split("_")
            pp_method = name[0]
            metric = name[1] if name[1] == "w2v" else "npmi"
            scores.append([metric, pp_method, topic_num, read_scores(open(file), topic_num)])
    return pd.DataFrame.from_records(scores, columns=["metric", "pp_method", "topic_num", "scores"])


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
    plot_df = copy.deepcopy(stat_df)
    lda_plot_df = copy.deepcopy(lda_df)
    for name in columns:
        plot_df[name] = plot_df[name].apply(take_mean)
        lda_plot_df[name] = lda_plot_df[name].apply(take_mean)
    names = [f"PP-{n}" for n in sorted(pps, reverse=True)] + ["Without PP"]
    for metric, group in plot_df.groupby("metric"):
        for topn in columns:
            fig = go.Figure()
            rows = lda_plot_df[lda_plot_df["metric"] == metric]
            kwargs = dict(marker_symbol="triangle-up", marker_size=8, line=dict(color="#19D3F3", width=2,
                                                                                dash=dashstyle[0]))
            fig.add_trace(go.Scatter(x=rows["topic_num"], y=rows[topn], name="LDA", **kwargs))
            for row, color, marker, mn, dash in zip(plot_rows, colors, markers, names, dashstyle[1:]):
                rows = group[group["pp_method"] == row]
                kwargs = dict(marker_symbol=marker, marker_size=8, line=dict(color=color, width=2, dash=dash))
                fig.add_trace(go.Scatter(x=rows["topic_num"], y=rows[topn], name=mn, **kwargs))
            # title = f"{topn} {metric} Coherence Scores for {task} task"
            fig.update_layout(xaxis_title='#Topic', template="plotly_white", width=1000, height=400,
                              yaxis_title=f"Mean {metric.upper()}")
            # fig.update_yaxes(range=[0, 0.6])
            os.makedirs(image_saved_dir / task, exist_ok=True)
            fig.write_image(image_saved_dir / task / f"{metric}_{topn}.png")
    print(f"Coherence scores plot saved in {image_saved_dir / task}")


def final_plot(stat_df, lda_df, image_saved_dir, percents=None, post_process=None):
    if percents is None:
        percents = [10, 30, 50, 100]
    if post_process is None:
        post_process = [100]
    plot_rows = [f"PP{i}" for i in sorted(post_process, reverse=True)] + ["original"]  # from upper to lower
    plot_df = copy.deepcopy(stat_df)
    lda_plot_df = copy.deepcopy(lda_df)
    columns = [f"Top-{p}%" for p in percents]
    for name in columns:
        plot_df[name] = plot_df[name].apply(take_mean)
        lda_plot_df[name] = lda_plot_df[name].apply(take_mean)

    names = [f"PP-{n}" for n in pps] + ["Without PP"]
    fig = go.Figure()
    for metric, group in plot_df.groupby("metric"):
        for topn in columns:
            rows = lda_plot_df[lda_plot_df["metric"] == metric]
            kwargs = dict(marker_symbol="triangle-up", marker_size=8, line=dict(color="#19D3F3", width=2))
            fig.add_trace(go.Scatter(x=rows["topic_num"], y=rows[topn], name=f"LDA({metric})", **kwargs))
            for row, color, marker, mn in zip(plot_rows, colors, markers, names):
                rows = group[group["pp_method"] == row]
                kwargs = dict(marker_symbol=marker, marker_size=8, line=dict(color=color, width=2))
                fig.add_trace(go.Scatter(x=rows["topic_num"], y=rows[topn], name=f"{mn}({metric})", **kwargs))
            fig.update_layout(xaxis_title='#Topic', template="plotly_white", width=1000, height=500,
                              yaxis_title=f"Topic Coherence Scores")
            fig.update_yaxes(range=[0, 0.6])
    os.makedirs(image_saved_dir / task, exist_ok=True)
    fig.write_image(image_saved_dir / f"{task}_final.png")
    print(f"Final combined coherence scores plot saved in {image_saved_dir / f'{task}_final.png'}")


def plot_box(topic_scores_df, lda_scores_df, image_saved_dir):
    plot_df = copy.deepcopy(topic_scores_df)
    lda_df = copy.deepcopy(lda_scores_df)
    plot_rows = ["original"] + [f"PP{i}" for i in pps]  # from left to right
    topic_scores_all, lda_scores_all = defaultdict(lambda: defaultdict(lambda: [])), defaultdict(lambda: [])
    plot_df = plot_df.drop_duplicates(subset=["metric", "pp_method", "topic_num"])
    lda_df = lda_df.drop_duplicates(subset=["metric", "topic_num"])
    box_kwargs = dict(jitter=0.4, whiskerwidth=0.2, marker=dict(size=1, color='rgb(0, 0, 0)'),
                      line=dict(width=1), boxpoints="all")
    for (metric, topic_num), group in plot_df.groupby(["metric", "topic_num"]):
        fig = go.Figure()
        for name, color in zip(plot_rows, colors):
            scores = np.stack(group[group["pp_method"] == name]["scores"].values).flatten()
            box_kwargs["fillcolor"] = color
            fig.add_trace(go.Box(y=scores, name=name, **box_kwargs))
            topic_scores_all[metric][name].extend(scores)
        lda_rows = lda_df[(lda_df["metric"] == metric) & (lda_df["topic_num"] == topic_num)]
        lda_score = np.stack(lda_rows["scores"].values).flatten()
        lda_scores_all[metric].extend(lda_score)
        box_kwargs["fillcolor"] = colors[-1]
        fig.add_trace(go.Box(y=lda_score, name="LDA", **box_kwargs))
        # title = f"Box plot of {metric} scores for {topic_num} topics"
        fig.update_layout(template="plotly_white", width=1000, height=500, yaxis_title=f"{metric.upper()} score")
        os.makedirs(image_saved_dir / task, exist_ok=True)
        fig.write_image(image_saved_dir / task / f"{metric}_{topic_num}.png")
    for metric, scores in topic_scores_all.items():
        fig = go.Figure()
        for ((name, score), color) in zip(scores.items(), colors):
            box_kwargs["fillcolor"] = color
            fig.add_trace(go.Box(y=score, name=name, **box_kwargs))
        box_kwargs["fillcolor"] = colors[-1]
        fig.add_trace(go.Box(y=lda_scores_all[metric], name="LDA", **box_kwargs))
        title = f"Box plot of {metric} scores for all topics"
        fig.update_layout(template="plotly_white", width=1000, height=500, yaxis_title=f"{metric.upper()} Score")
        fig.write_image(image_saved_dir / task / f"{metric}_all.png")
    print("save box plots to", image_saved_dir)


def coherence_table(stat_df, lda_df, table_saved_dir, percents=None):
    if percents is None:
        percents = [10, 30, 50, 100]
    plot_rows = ["original"] + [f"PP{i}" for i in pps]  # from lower to upper
    for (metric, topic_num), group in stat_df.groupby(["metric", "topic_num"]):
        records = []
        for p in percents:
            line = [f"Top-{p}%"]
            for name in plot_rows:
                rows = group[group["pp_method"] == name]
                line.append(rows[f"Top-{p}%"].values[0])
            lda_rows = lda_df[(lda_df["metric"] == metric) & (lda_df["topic_num"] == topic_num)]
            line.append(lda_rows[line[0]].values[0])
            records.append(line)
        table = pd.DataFrame.from_records(records, columns=["Percentage"] + plot_rows + ["LDA"])
        os.makedirs(table_saved_dir / task, exist_ok=True)
        table.to_csv(table_saved_dir / task / f"{metric}_{topic_num}_table.csv", index=False)
    print("save tables to", table_saved_dir)


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    batm_coherence_dir = cmd_args.get("batm_coherence_dir", None)  # the directory of topic list
    lda_coherence_dir = cmd_args.get("lda_coherence_dir", None)  # the directory of lda terms
    if batm_coherence_dir is None or lda_coherence_dir is None:
        raise ValueError("The directory of batm model and lda coherence scores should be provided.")
    saved_dir = Path(cmd_args.get("save_dir", f"{get_project_root()}/saved/stat/coherence_score"))
    plot_dir = Path(cmd_args.get("plot_dir", f"{get_project_root()}/saved/plot_images/percents_coherence"))
    box_dir = Path(cmd_args.get("box_dir", f"{get_project_root()}/saved/plot_images/coherence_box_plot"))
    coherence_table_dir = Path(cmd_args.get("coherence_table_dir", f"{get_project_root()}/saved/stat/percents_table"))
    os.makedirs(saved_dir, exist_ok=True)
    os.makedirs(coherence_table_dir, exist_ok=True)
    pps = [10, 60]  # the number of post-processing
    percentages = [10, 30, 50, 80, 100]  # the percentage of top topics
    lda_scores = sort_df(load_lda_scores(lda_coherence_dir), "topic_num")
    lda_percents_scores_df = cal_group_scores(lda_scores, ["metric", "topic_num"], percentages)
    lda_percents_scores_df = sort_df(lda_percents_scores_df, "topic_num")  # sort by topic number
    lda_percents_scores_df.to_csv(saved_dir / "lda_percents_scores.csv", index=False)
    print("save the lda coherence scores on path:", saved_dir / "lda_percents_scores.csv")
    batm_coherence_dir = Path(batm_coherence_dir)
    task = batm_coherence_dir.name.split("_")[0]
    batm_scores = sort_df(load_batm_scores(batm_coherence_dir), "topic_num")
    batm_percents_scores_df = cal_group_scores(batm_scores, ["pp_method", "metric", "topic_num"], percentages)
    batm_percents_scores_df = sort_df(batm_percents_scores_df, "topic_num")  # sort by topic number
    batm_percents_scores_df.to_csv(saved_dir / f"{task}_batm_percents_scores.csv", index=False)
    print("save the batm coherence scores on path:", saved_dir / f"{task}_batm_percents_scores.csv")
    coherence_percents_plot(batm_percents_scores_df, lda_percents_scores_df, plot_dir, percents=percentages)
    final_plot(batm_percents_scores_df, lda_percents_scores_df, plot_dir, percents=[100])
    plot_box(batm_scores, lda_scores, box_dir)
    coherence_table(batm_percents_scores_df, lda_percents_scores_df, coherence_table_dir, percents=percentages)
