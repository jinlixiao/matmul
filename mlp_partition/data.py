import torch
from torch.utils.data import Dataset

class RandomDataset(Dataset):
    def __init__(self, data_size, input_size, output_size, seed=0):
        self.data_size = data_size
        self.input_size = input_size
        self.output_size = output_size

        torch.manual_seed(seed)
        self.data_set = torch.randn(self.data_size, self.input_size)
        self.labels = torch.randn(self.data_size, self.output_size)
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        inputs = self.data_set[idx]
        output = self.labels[idx]
        return inputs, output
