from modules.dataset.base_rs_dataset import MindRSDataset
from modules.utils import Tokenizer


class GraphDataset(MindRSDataset):
    def __init__(self, tokenizer: Tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        pass
