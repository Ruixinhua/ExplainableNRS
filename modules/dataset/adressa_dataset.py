from torch.utils.data import Dataset


class AdressaDataset(Dataset):

    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        text = self.tokenizer.encode_plus(text, max_length=self.max_length, pad_to_max_length=True, truncation=True, return_tensors='pt')
        return text
