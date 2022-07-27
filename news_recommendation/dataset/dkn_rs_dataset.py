from news_recommendation.dataset.base_rs_dataset import MindRSDataset
from news_recommendation.utils import Tokenizer
from news_recommendation.utils import load_news_feature, load_utils_file


class DKNRSDataset(MindRSDataset):
    def __init__(self, tokenizer: Tokenizer, **kwargs):
        # TODO: DKN RS Dataset
        super().__init__(tokenizer, **kwargs)
        news_file = load_utils_file(**kwargs)["news_feature_file"]
        self.nid2index, self.news_matrix = load_news_feature(news_file, **kwargs)
