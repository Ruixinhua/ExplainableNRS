from modules.dataset.base_rs_dataset import NewsRecDataset
from modules.utils import Tokenizer


class GraphDataset(NewsRecDataset):
    def __init__(self, tokenizer: Tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        pass
