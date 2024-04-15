import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class BaseDataset(Dataset):
    def __init__(self, split, config):
        assert split in ['train', 'val', 'test']

        root = config.Path.data
        data_path = os.path.join(root, split + ".csv")
        df = pd.read_csv(str(data_path))

        self.data = []

        if split in ['train', 'val']:
            for _, row in df.iterrows():
                item = dict()
                item["text"] = row["text"]
                item["titles"] = row["titles"]
                self.data.append(item)
        else:
            for _, row in df.iterrows():
                item = dict()
                item["text"] = row["text"]
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def base_dataloader(batch_size, shuffle, split, config):
    dataset = BaseDataset(split, config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
