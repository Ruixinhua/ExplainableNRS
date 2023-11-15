import math
import pandas as pd
import numpy as np
import modules.utils.metric_utils as module_metric
from collections import defaultdict
from tqdm import tqdm

tqdm.pandas()
np.random.seed(42)

metric_funcs = [
    getattr(module_metric, met)
    for met in ["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"]
]


def load_all_data(mind_type):
    root_dir = f"../dataset/MIND/{mind_type}"
    test_df = pd.read_csv(
        f"{root_dir}/test/behaviors.tsv",
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "label"],
    )
    test_df["subset"] = "test"
    train_df = pd.read_csv(
        f"{root_dir}/train/behaviors.tsv",
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "label"],
    )
    valid_df = pd.read_csv(
        f"{root_dir}/valid/behaviors.tsv",
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "label"],
    )
    news_df = pd.read_csv(f"{root_dir}/news.csv")
    news_topic = dict(zip(news_df.news_id.tolist(), news_df.category.tolist()))
    all_df = pd.concat([train_df, valid_df, test_df])
    return all_df, news_topic


def load_labels(all_df):
    return (
        all_df[all_df["subset"] == "test"]
        .label.apply(lambda x: np.array([int(i.split("-")[1]) for i in x.split(" ")]))
        .tolist()
    )


def load_candidates(all_df):
    return (
        all_df[all_df["subset"] == "test"]
        .label.apply(lambda x: np.array([i.split("-")[0] for i in x.split(" ")]))
        .tolist()
    )


def compute_result(mn, mt, ls, ps):
    result = {"model": mn, "mind_type": mt}
    metrics = defaultdict(list)
    for l, p in zip(ls, ps):
        for func in metric_funcs:
            metrics[func.__name__].append(round(func(l, p), 5))

    for k, v in metrics.items():
        result[k] = np.round(np.mean(v) * 100, 2)
    return result


def compute_random_result(all_df, mind_type):
    labels = load_labels(all_df)
    return compute_result(
        "Random", mind_type, labels, [np.random.rand(len(l)) for l in labels]
    )


def cal_count(all_df):
    count = defaultdict(int)
    # stat the popularity of history clicked news articles
    for h in all_df.history:
        if type(h) == float:
            continue
        for n in h.split(" "):
            count[n] += 1
    for h in all_df.label:
        if type(h) == float:
            continue
        for n in h.split(" "):
            count[n.split("-")[0]] += 1
    return count


def cal_pop(all_df):
    count = cal_count(all_df)
    candidates = load_candidates(all_df)
    return [np.array([count[n] for n in c]) for c in candidates]


def compute_most_pop_result(all_df, mind_type):
    labels = load_labels(all_df)
    return compute_result("MostPop", mind_type, labels, cal_pop(all_df))


def compute_time_weight(time_gap, term=20):
    lam = 1 / term
    weight = math.exp(-lam * time_gap)
    return weight


def compute_time_gap(latest, old):
    return (latest - old).total_seconds() / 60


def convert_timestamp(timestamp_str, delta="1hour"):
    timestamp = pd.to_datetime(timestamp_str)  # Convert to datetime
    total_minutes = (timestamp - pd.Timestamp("1970-01-01")) // pd.Timedelta(
        delta
    )  # Convert to total minutes
    return total_minutes


def compute_time_delta(all_df, delta="1hour"):
    return all_df.time.progress_apply(lambda x: convert_timestamp(x, delta))


def cal_recency_count(all_df, term=20, delta="time_in_hour"):
    recency_count = defaultdict(lambda: [0, 0])
    count = defaultdict(int)
    max_time = all_df[delta].max()
    all_df["time_gap"] = all_df[delta].apply(lambda x: max_time - x)
    all_df["time_weight"] = all_df[delta].apply(
        lambda x: compute_time_weight(max_time - x, term=term)
    )
    for h, w in tqdm(
        zip(all_df.history.tolist(), all_df.time_weight.tolist()),
        total=len(all_df),
        disable=True,
    ):
        if type(h) == float:
            continue
        for n in h.split(" "):
            recency_count[n][0] = w if w > recency_count[n][0] else recency_count[n][0]
            recency_count[n][1] += 1
            count[n] = recency_count[n][0] * recency_count[n][1]
    for h, w in tqdm(
        zip(all_df.label.tolist(), all_df.time_weight.tolist()),
        total=len(all_df),
        disable=True,
    ):
        if type(h) == float:
            continue
        for n in h.split(" "):
            n = n.split("-")[0]
            recency_count[n][0] = w if w > recency_count[n][0] else recency_count[n][0]
            recency_count[n][1] += 1
            count[n] = recency_count[n][0] * recency_count[n][1]
    return count


def compute_recency_pop_result(all_df, mind_type, term, delta="time_in_hour"):
    labels = load_labels(all_df)
    count = cal_recency_count(all_df, term=term, delta=delta)
    pop_list = [np.array([count[n] for n in c]) for c in load_candidates(all_df)]
    return compute_result("recencyPop", mind_type, labels, pop_list)


def compute_topic_pop_result(all_df, mind_type, topic_map):
    pop_count = cal_count(all_df)
    max_pop = max(pop_count.values())
    test_df = all_df[all_df["subset"] == "test"]
    topic_pop = []
    for his, cans in test_df[["history", "label"]].values:
        cans = [n.split("-")[0] for n in cans.split(" ")]
        if type(his) == float:
            pop = [pop_count[n] for n in cans]
        else:
            his = [n for n in his.split(" ")]
            his_topic = [topic_map[n] for n in his]
            cans_topic = [topic_map[n] for n in cans]
            pop = []
            for i, n in enumerate(cans):
                if cans_topic[i] in his_topic:
                    pop.append(max(pop_count[n] * 10, max_pop))
                else:
                    pop.append(pop_count[n])
        topic_pop.append(np.array(pop))
    return compute_result("topicPop", mind_type, load_labels(all_df), topic_pop)


def get_mean_std(values, r: int = 2):
    return (
        f"{np.round(np.mean(values), r)}" + "\u00B1" + f"{np.round(np.std(values), r)}"
    )


def cal_mean_std(df):
    return {met.__name__: get_mean_std(df[met.__name__].values) for met in metric_funcs}


if __name__ == "__main__":
    small_df, small_topic = load_all_data("small")
    large_df, large_topic = load_all_data("large")

    random_results_small = [compute_random_result(small_df, "small") for _ in range(5)]
    print(cal_mean_std(pd.DataFrame(random_results_small)))

    random_results_large = [compute_random_result(large_df, "large") for _ in range(5)]
    print(cal_mean_std(pd.DataFrame(random_results_large)))

    most_pop_result_small = compute_most_pop_result(small_df, "small")
    print(most_pop_result_small)
    most_pop_result_large = compute_most_pop_result(large_df, "large")
    print(most_pop_result_large)

    small_df["time_in_hour"] = compute_time_delta(small_df, delta="1hour")
    # small_df["time_in_day"] = compute_time_delta(small_df, delta="1day")

    term = 144
    recency_pop_result_small = compute_recency_pop_result(
        small_df, "small", term, delta="time_in_hour"
    )
    print(recency_pop_result_small)

    large_df["time_in_hour"] = compute_time_delta(large_df, delta="1hour")
    # large_df["time_in_day"] = compute_time_delta(large_df, delta="1day")
    term = 105
    recency_pop_result_large = compute_recency_pop_result(
        large_df, "large", term, delta="time_in_hour"
    )
    print(recency_pop_result_large)

    topic_pop_result_small = compute_topic_pop_result(small_df, "small", small_topic)
    print(topic_pop_result_small)

    topic_pop_result_large = compute_topic_pop_result(large_df, "large", large_topic)
    print(topic_pop_result_large)
