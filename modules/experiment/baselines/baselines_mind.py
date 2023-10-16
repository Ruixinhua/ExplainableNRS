import os
import tensorflow as tf

from datetime import datetime

from pathlib import Path

from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.datasets.mind import (
    read_clickhistory,
    get_train_input,
    get_valid_input,
    get_user_history,
    get_words_and_entities,
    generate_embeddings,
)
from modules.experiment.baselines import (
    get_model_class,
    get_iterator_class,
    get_train_valid_path,
    get_yaml_path,
)
from modules.config.configuration import Configuration
from modules.config.config_utils import set_seed, load_cmd_line
from modules.utils import get_project_root, check_existing

tf.get_logger().setLevel("ERROR")  # only show error messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == "__main__":
    # setup arguments used to run baseline models
    cmd_args = load_cmd_line()
    timestamp = datetime.now().strftime(r"%m%d_%H%M%S")
    if "nc" in cmd_args["task"].lower():
        config = Configuration()
    else:
        config_file = (
            Path(get_project_root()) / "modules" / "config" / "mind_rs_default.json"
        )
        config = Configuration(config_file=config_file)
    saved_dir_name = (
        cmd_args["saved_dir_name"] if "saved_dir_name" in cmd_args else "performance"
    )
    data_path = Path(
        config.get("data_dir", f"{get_project_root()}/dataset/MIND_Original")
    )
    model_name = config.get("arch_type", "DKN")
    MIND_SIZE = config.get("subset_type", "small")
    iterator = get_iterator_class(model_name)
    set_seed(config.get("seed", 42))
    yaml_file = get_yaml_path(model_name, data_path, MIND_SIZE)
    if model_name == "DKN":
        dkn_utils = data_path / "dkn_utils"
        check_existing(dkn_utils)
        train_file = dkn_utils / "train_mind.txt"
        valid_file = dkn_utils / "valid_mind.txt"
        word_embedding_dim = config.get("word_embedding_dim", 100)
        params = {
            "user_history_file": str(dkn_utils / "user_history.txt"),
            "news_feature_file": str(dkn_utils / "doc_feature.txt"),
            "wordEmb_file": str(
                dkn_utils / f"word_embeddings_5w_{word_embedding_dim}.npy"
            ),
            "entityEmb_file": str(
                dkn_utils / f"entity_embeddings_5w_{word_embedding_dim}.npy"
            ),
        }
        train_path, valid_path = get_train_valid_path(data_path, MIND_SIZE)
        train_session, train_history = read_clickhistory(train_path, "behaviors.tsv")
        valid_session, valid_history = read_clickhistory(valid_path, "behaviors.tsv")
        if not train_file.exists():
            get_train_input(train_session, str(train_file))
        if not valid_file.exists():
            get_valid_input(valid_session, str(valid_file))
        if not check_existing(params["user_history_file"], mkdir=False):
            get_user_history(train_history, valid_history, params["user_history_file"])
        train_news = os.path.join(train_path, "news.tsv")
        valid_news = os.path.join(valid_path, "news.tsv")
        news_words, news_entities = get_words_and_entities(train_news, valid_news)
        train_entities = os.path.join(train_path, "entity_embedding.vec")
        valid_entities = os.path.join(valid_path, "entity_embedding.vec")
        is_exist = (
            Path(params["wordEmb_file"]).exists()
            and Path(params["entityEmb_file"]).exists()
        )
        if not is_exist:
            generate_embeddings(
                str(dkn_utils),
                news_words,
                news_entities,
                train_entities,
                valid_entities,
                max_sentence=10,
                word_embedding_dim=word_embedding_dim,
            )
        hparams = prepare_hparams(
            yaml_file,
            epochs=config.get("epochs", 10),
            history_size=config.get("history_size", 50),
            batch_size=config.get("batch_size", 32),
            **params,
        )
        model = get_model_class(model_name)(hparams, iterator)
        fit_files = [train_file, valid_file]
        model.fit(train_file, valid_file)
        res = model.run_eval(valid_file)
    else:
        train_news_file = data_path / "train" / "news.tsv"
        train_behaviors_file = data_path / "train" / "behaviors.tsv"
        valid_news_file = data_path / "valid" / "news.tsv"
        valid_behaviors_file = data_path / "valid" / "behaviors.tsv"
        hparams = prepare_hparams(
            yaml_file,
            wordEmb_file=data_path / "utils" / f"embedding.npy",
            wordDict_file=data_path / "utils" / "word_dict.pkl",
            userDict_file=data_path / "utils" / "uid2index.pkl",
            epochs=config.get("epochs", 10),
            history_size=config.get("history_size", 50),
            batch_size=config.get("batch_size", 32),
        )
        model = get_model_class(model_name)(hparams, iterator)
        model.fit(
            train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file
        )
        res = model.run_eval(valid_news_file, valid_behaviors_file)
    print(res)
