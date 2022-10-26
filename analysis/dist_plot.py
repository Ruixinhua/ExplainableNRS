import os

import pandas as pd
from collections import defaultdict
from pathlib import Path
import plotly.graph_objects as go

from modules.config import load_cmd_line


def load_dist(td):
    dist = defaultdict(lambda: [])
    for file in os.scandir(td):
        if file.name == "topic_list.txt":
            continue
        with open(file) as r:
            scores = []
            for line in r:
                if line.startswith("Average"):
                    break
                scores.append(eval(line.split(":")[0]))
            dist[f"{file.name.split('_')[2]}_{len(scores)}"] = scores
    return dist


def cal_per_scores(scores, p):
    num = int(p * len(scores) / 100)
    return round(sum(scores[:num]) / num, 4)


def percentage_scores(dist, ps=None, keys=None):
    if ps is None:
        ps = [10, 30, 50, 100]
    scores = []
    columns = ["Identifier"] + [f"Top-{p}%" for p in ps[:-1]] + ["All"]
    keys = dist.keys() if keys is None else keys
    for key in keys:
        if key not in dist:
            continue
        scores.append([key] + [cal_per_scores(dist[key], p) for p in ps])
    return pd.DataFrame.from_records(scores, columns=columns)


def plot_box(topic_dir, num):
    plot_keys = [f"original_{num}"] + [f"PP{p}_{num}" for p in pps]
    plot_names = ["Without PP"] + [f"Post-Process-{n}" for n in pps]
    topic_scores = load_dist(topic_dir)
    topic_dist = percentage_scores(topic_scores, ps=percentages, keys=plot_keys)
    fig = go.Figure()
    for pk, name in zip(plot_keys, plot_names):
        if pk not in topic_scores:
            continue
        fig.add_trace(go.Box(y=topic_scores[pk], name=name))
    fig.add_trace(go.Box(y=lda_scores[f"lda_{num}"], name="LDA"))
    fig.update_layout(title=f"Box plot of NPMI scores with {num} topics", template="plotly_white",
                      width=1000, height=500, yaxis_title="NPMI score")
    fig.show()
    fig.write_image(f"images/npmi_box_compare_{num}.png")
    return topic_dist.set_index("Identifier").T


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    topic_dir = cmd_args.get("topic_dir", None)  # the directory of topic list
    pps = [10, 30, 60, 100]  # the number of post-processing
    percentages = [10, 30, 50, 80, 100]  # the percentage of top topics
