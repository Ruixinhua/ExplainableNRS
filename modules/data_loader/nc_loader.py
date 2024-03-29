from pathlib import Path
from torch.utils.data import DataLoader
from modules.base.nc_dataset import NCDatasetBert, NCDataset
from modules.utils import load_dataset_df, load_word_dict, load_embeddings, split_df, get_project_root


class NewsDataLoader:
    def load_dataset(self, df):
        from modules.config.default_config import TEST_CONFIGS
        pretrained_models = TEST_CONFIGS["bert_embedding"]
        if self.embedding_type in pretrained_models:
            dataset = NCDatasetBert(texts=df["data"].values.tolist(), labels=df["category"].values.tolist(),
                                    label_dict=self.label_dict, max_length=self.max_length,
                                    embedding_type=self.embedding_type)
            if self.embedding_type == "transfo-xl-wt103":
                self.word_dict = dataset.tokenizer.sym2idx
            else:
                self.word_dict = dataset.tokenizer.vocab
        elif self.embedding_type in ["glove", "init"]:
            # if we use glove embedding, then we skip the unknown words
            dataset = NCDataset(df["data"].values.tolist(), df["category"].values.tolist(), self.label_dict,
                                self.max_length, self.word_dict, self.method)
        else:
            raise ValueError(f"Embedding type should be one of {','.join(pretrained_models)} or glove and init")
        return dataset

    def __init__(self, batch_size=32, shuffle=True, num_workers=1, **kwargs):
        self.method = kwargs.get("tokenized_method", "keep_all")
        self.set_name = kwargs.get("dataset_name", "MIND15")
        self.max_length, self.embedding_type = kwargs.get("max_length", 512), kwargs.get("embedding_type", "glove")
        self.data_dir = kwargs.get("data_dir", Path(get_project_root(), "dataset"))
        self.data_path = kwargs.get("data_path", Path(self.data_dir) / f"MIND/news_classification/{self.set_name}.csv")
        kwargs["data_path"] = self.data_path
        df, self.label_dict = load_dataset_df(**kwargs)
        if "split" not in df:
            df = split_df(df, split_test=True)
            df.to_csv(self.data_path, index=False)
        # load index of training, validation and test set
        train_set, valid_set, test_set = df["split"] == "train", df["split"] == "valid", df["split"] == "test"
        if self.embedding_type in ["glove", "init"]:
            # setup word dictionary for glove or init embedding
            self.word_dict = load_word_dict(**kwargs)
        if self.embedding_type == "glove":
            self.embeds = load_embeddings(**kwargs)
        self.init_params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers}
        # initialize train loader
        self.train_loader = DataLoader(self.load_dataset(df[train_set]), **self.init_params)
        self.init_params["shuffle"] = False  # disable shuffle for validation and test set
        # initialize validation loader
        self.valid_loader = DataLoader(self.load_dataset(df[valid_set]), **self.init_params)
        # initialize test loader
        self.test_loader = DataLoader(self.load_dataset(df[test_set]), **self.init_params)
        # initialize all loader
        self.all_loader = DataLoader(self.load_dataset(df), **self.init_params)
