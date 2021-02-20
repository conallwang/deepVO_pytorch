from torch.utils.data import Dataset


class DeepVODataset(Dataset):
    def __init__(self, seq=2, interval=1):
        super().__init__()

        self.seq = 2
        self.interval = 1

    def load_data():
        pass
    
    def __getitem__(self, index):
        pass

    def __len__(self):
        pass