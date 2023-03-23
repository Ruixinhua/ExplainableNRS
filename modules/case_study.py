import os
import pandas as pd
import numpy as np
import torch
import re
import string

from collections import defaultdict
from pathlib import Path

from modules.config import load_cmd_line, COLOR_LIST
from modules.utils import read_json, clean_df, get_project_root, get_topn


def acquire_news_info(imp_index, item_index, prefix="history"):
    news_index = weight_dict[f"{prefix}_index"][imp_index][item_index]
    news_id = f"<i>{index2nid[news_index]}</i>"  # display in table
    news = news_df[news_df["news_index"] == news_index]
    weights = np.round(weight_dict[f"{prefix}_weight"][imp_index][item_index], 6)
    weights[topics_exclude] = 0  # mask topic weights with low topic coherence
    topic_index_sorted = get_topn(weights, show_topic_num)  # most important top 5 topics
    for j in topic_index_sorted:
        if j not in involved_topic_dict:
            involved_topic_dict[j] = len(involved_topic_dict)
            topic_stat[j] = 1
        topic_stat[j] += 1
    top_topics = u"".join([f"<span style='color:{COLOR_LIST[involved_topic_dict[k]]}'>Topic-{k}</span><br>" for j, k in
                           enumerate(topic_index_sorted)])  # display in table
    topics = weight_dict[f"{prefix}_topic_weight"][i][item_index]
    tokenized_text = tokenized_news[news_index].split()
    # print(f"{index},{news_index}: {len(tokenized_text)}")
    terms_indices = {t: ti for ti, t in enumerate(tokenized_text[:100]) if t in pp_word_dct}
    topic_terms100 = topics[:, list(terms_indices.values())]
    terms100 = list(terms_indices.keys())
    top_topic_terms = defaultdict(lambda: [])
    for ti in topic_index_sorted:
        for t in get_topn(topic_terms100[ti], show_term_num):
            top_topic_terms[terms100[t]].append(
                (ti, topic_terms100[ti][t]))  # acquire topic number and score (multiple topics)
    title, abstract, body = news["title"].values[0], news["abstract"].values[0], news["body"].values[0]
    title = title + "." if not title.endswith(".") else title
    abstract = abstract + "." if not abstract.endswith(".") else abstract
    pat = re.compile(r"[\w]+|[.,!?;|]")
    news_terms = title + " " + abstract + " " + body
    news_terms = pat.findall(news_terms.strip("\n"))
    news_contents = u"<span>"
    k = 0
    for ti, s in enumerate(news_terms):
        if s.lower() == tokenized_text[k]:
            k += 1
        if s.lower() in top_topic_terms:
            tn_score = top_topic_terms[s.lower()]
            best_topic = tn_score[0][0]
            best_score = tn_score[0][1]
            for tn, score in tn_score:
                if score > best_score:
                    best_score = score
                    best_topic = tn
            # topic_index = involved_topic_dict[best_topic]
            s = f"<span style='color:{COLOR_LIST[involved_topic_dict[best_topic]]}'>{s}</span>"
        if ti == 0 or s in string.punctuation:
            s = s
        else:
            s = f" {s}"
        news_contents += s
        if k >= 100 or k >= len(tokenized_text):
            news_contents += "...</span>"
            break
    top_terms = "<br>".join(
        [",".join([f"{terms100[t]}" for t in get_topn(topic_terms100[ti], 10)]) for ti in topic_index_sorted]
    )
    topic_weight_info = "<br>".join([
        f"<span style='color:{COLOR_LIST[involved_topic_dict[ti]]}'>Topic-{ti}({str(weights[ti])})</span>"
        for j, ti in enumerate(topic_index_sorted)
    ])
    topic_terms = "<br>".join([",".join(topic_terms_list[ti]) for ti in topic_index_sorted])
    involved_topics.append([news_id, topic_weight_info, topic_terms, top_terms])
    return [news_index, news_id, news_contents, news["category"].values[0], news["subvert"].values[0], top_topics]


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    topic_num = cmd_args.get('topic_num', 300)
    show_topic_num = cmd_args.get('show_topic_num', 5)
    show_term_num = cmd_args.get('show_term_num', 5)
    show_candidate_num = cmd_args.get('show_candidate_num', 5)
    show_history_num = cmd_args.get('show_history_num', 5)
    top_percent = cmd_args.get('top_percent', 0.1)
    correct_only = cmd_args.get('correct_only', True)
    out_name = cmd_args.get('out_name', f"TN{str(topic_num)}")
    project_root = Path(get_project_root())

    saved_dir = project_root / "saved"
    weight_dir = saved_dir / "models" / "MIND15" / "RS_BATM_base_att_small_base_tanh_hd30_20221107-223130" / "weight"
    weight_dir = Path(cmd_args.get('weight_dir', weight_dir))
    weight_dict = torch.load(weight_dir / f"{topic_num}.pt")
    impression_index = weight_dict["impression_index"]
    selected_index = cmd_args.get("selected_imp", None)
    if selected_index:
        indices = [impression_index.index(i) for i in selected_index]
        weight_dict = {k: [v[i] for i in indices] for k, v in weight_dict.items()}
    pp_word_dct = read_json(project_root / "dataset/utils/word_dict/post_process/PP100.json")
    default_topic_file = Path(saved_dir, "models", "MIND15", "RS_BATM_base_att_small_base_tanh_hd30_20221107-223130",
                              "topics", "topics_4_300_unsorted", "PP100_c_npmi_0.1392.txt")
    # r"
    topics_file = cmd_args.get("topic_file", default_topic_file)
    with open(topics_file) as reader:  # load unsorted topics
        topic_list = [next(reader).split(":") for _ in range(int(topic_num))]
        topic_terms_list = [topics[1].split() for topics in topic_list]
        topic_scores_list = [eval(topics[0]) for topics in topic_list]
    top_topics_indices = get_topn(topic_scores_list, int(len(topic_scores_list) * top_percent))
    topics_exclude = np.ones_like(topic_scores_list, dtype=bool)
    topics_exclude[top_topics_indices] = False
    mind_dir = project_root / Path(r"dataset/MIND/small")
    news_file = mind_dir / "news.csv"  # define news file path
    # columns = ["news_id", "category", "subvert", "title", "abstract", "url", "entity", "ab_entity"]
    # initial data of corresponding news attributes, such as: title, entity, vert, subvert, abstract
    # news_df = pd.read_table(news_file, header=None, names=columns)
    news_df = pd.read_csv(news_file)
    nid2index = read_json(r"C:\Users\Rui\Documents\Explainable_AI\explainable_nrs\dataset\utils\MIND_nid_small.json")
    index2nid = dict(zip(nid2index.values(), nid2index.keys()))
    # article_path = subset_dir / "msn.json"
    # if os.path.exists(article_path):
    #     articles = read_json(article_path)
    #     news_df["body"] = news_df.news_id.apply(lambda nid: " ".join(articles[nid]) if nid in articles else "")
    news_df = clean_df(news_df)
    news_df["docs"] = news_df["title"] + " " + news_df["abstract"] + " " + news_df["body"]
    tokenized_news_path = project_root / Path(r"dataset/data/MIND_small_original.csv")
    tokenized_news = [""]
    tokenized_news.extend(pd.read_csv(tokenized_news_path)["tokenized_text"].tolist())
    # news_df = pd.merge(news_df, tokenized_news, on="news_id", how="left").fillna("")
    news_df["news_index"] = news_df.news_id.apply(lambda nid: nid2index[nid])
    news_columns = ["news_index", "news_id", "news_content", "category", "sub_category", "top_topics"]
    topic_columns = ["<b>News ID</b>", "<b>Topic No.(weight)</b>", "<b>Top-10 Topic Terms</b>",
                     "<b>Top-10 Topic Terms Among News</b>"]
    table_header = ["<b>News ID</b>", "<b>News Content</b>", "<b>Category</b>", "<b>Subcategory</b>",
                    "<b>Top Topics No.</b>"]
    history_header = table_header + ["<b>W(User-News)</b>"]
    candidate_header = table_header + ["<b>Clicked</b>", "<b>Rank</b>"]
    correct_indices = [index for index, result in enumerate(weight_dict["results"]) if result["group_auc"] >= 1]
    all_indices = [index for index, result in enumerate(weight_dict["results"])]
    indices = correct_indices if correct_only else all_indices
    for num, i in enumerate(indices):
        pred_score = np.exp(weight_dict["pred_score"][i]) / sum(np.exp(weight_dict["pred_score"][i]))
        history_case = []
        history_index_sorted = get_topn(weight_dict["user_weight"][i], show_history_num)
        involved_topics = []
        involved_topic_dict = {}
        topic_stat = {}
        for index in history_index_sorted:
            if weight_dict["history_index"][i][index] not in index2nid:
                continue
            user_weight = weight_dict["user_weight"][i][index] / sum(
                weight_dict["user_weight"][i][history_index_sorted])
            history_case.append(acquire_news_info(i, index) + [user_weight])
        if len(history_case) == 0:
            continue
        history_case_df = pd.DataFrame.from_records(history_case, columns=news_columns + ["user_news_weight"])
        candidate_case = []
        positive_index = []
        negative_index = []
        for index, label in enumerate(weight_dict["label"][i]):
            if label:
                positive_index.append(index)
            else:
                negative_index.append(index)
        np.random.shuffle(negative_index)
        candidate_sample = get_topn(pred_score, show_candidate_num)
        # candidate_sample = positive_index + negative_index[:max((5 - len(positive_index)), 0)]
        # select top-n scores candidate news
        for rank, index in enumerate(candidate_sample):
            candidate_case.append(acquire_news_info(i, index, "candidate") + [weight_dict["label"][i][index], rank+1])
        candidate_case_df = pd.DataFrame.from_records(candidate_case, columns=news_columns + ["label", "rank"])
        category = candidate_case_df[candidate_case_df["rank"] == 1]["category"][0]
        involved_topics_df = pd.DataFrame.from_records(involved_topics, columns=topic_columns)
        # print(topic_stat)
        pd.set_option('colheader_justify', 'center')  # FOR TABLE <th>
        history_values = [history_case_df.news_id, history_case_df.news_content, history_case_df.category,
                          history_case_df.sub_category, history_case_df.top_topics, history_case_df.user_news_weight]
        # candidate_case_df["score"] = candidate_case_df.score.apply(lambda s: round(s, 4))
        candidate_values = [candidate_case_df.news_id, candidate_case_df.news_content, candidate_case_df.category,
                            candidate_case_df.sub_category, candidate_case_df.top_topics, candidate_case_df.label,
                            candidate_case_df["rank"]]
        history_df = pd.DataFrame.from_dict(dict(zip(history_header, history_values)))
        candidate_df = pd.DataFrame.from_dict(dict(zip(candidate_header, candidate_values)))
        css_style = """
        .mystyle, .topics{
                font-size: 11pt;
                font-family: Arial;
                border-collapse: collapse;
                border: 1px solid silver;
                text-align: center;
                /*table-layout: fixed;*/
                /* this keeps your columns with fixed with exactly the right width */
                /* table must have width set for fixed layout to work as expected */
                width: 100%;
            }

            .mystyle td, th {
                padding: 5px;
            }
            .topics td, th {
                padding: 5px;
            }
            .mystyle td:nth-child(1) {
                width: 5%;
            }
            .mystyle td:nth-child(2) {
                width: 60%;
                text-align: left;
            }
            .mystyle td:nth-child(3) {
                width: 5%;
            }
            .mystyle td:nth-child(4) {
                width: 5%;
            }
            .mystyle td:nth-child(5) {
                width: 10%;
            }
            .topics td:nth-child(1) {
                width: 10%;
            }
            .topics td:nth-child(2) {
                width: 20%;
            }
            .topics td:nth-child(3) {
                width: 35%;
                text-align: left;
            }
            .topics td:nth-child(4) {
                width: 35%;
                text-align: left;
            }
        """
        group_auc = round(weight_dict["results"][i]["group_auc"], 2)
        html_string = f'''
        <html>
          <head><title>TN{str(topic_num)}-{category}{impression_index}-{group_auc}</title></head>

            <style>
            /* includes alternating gray and white with on-hover color */
            {css_style}
            </style>
          <body>
          <h2>Case Study No.{num}
          <h3>User History Browsed News</h3>
            {history_df.to_html(classes='mystyle', escape=False, index=False)}
            <h3>Candidate News for the Current User</h3>
            {candidate_df.to_html(classes='mystyle', escape=False, index=False)}
            <h3> {involved_topics_df.to_html(classes='topics', escape=False, index=False)}</h3>
          </body>
        </html>.
        '''
        output_dir = saved_dir / f"case_study/{out_name}"
        os.makedirs(output_dir, exist_ok=True)
        # OUTPUT AN HTML FILE
        impression_index = weight_dict["impression_index"][i]
        with open(output_dir / f'TN{str(topic_num)}-{category}{impression_index}-{group_auc}.html', 'w') as f:
            f.write(html_string)
