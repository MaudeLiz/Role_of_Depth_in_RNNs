import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from splearn.datasets.base import load_data_sample
from splearn.tests.datasets.get_dataset_path import get_dataset_path
from lightning import LightningDataModule
import os


class PautomacDataset(Dataset):
    def __init__(self, automata_path):
        self.automata_path = automata_path
        self.data, self.vocab_size = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_data(self):
        # Load the automata data from the specified file
        data = load_data_sample(self.automata_path)
        vocab_size = data.nbL + 1
        data = data.data
        # convert data to torch tensor with int datatype
        data = torch.tensor(data, dtype=torch.int64) + 1

        return data, vocab_size


class PautomacDataModule(LightningDataModule):
    def __init__(self, dataset, batch_size=32, num_workers=0):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = dataset.vocab_size
        self.automata_path = dataset.automata_path
        self.batch_size = batch_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset,
            [0.8, 0.1, 0.1],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )




# Example usage
if __name__ == "__main__":
    automata_path = "/Users/michaelrizvi/data/PAutomaC-competition_sets/8.pautomac.train" 
    dataset = PautomacDataset(automata_path)

    data_module = PautomacDataModule(dataset, batch_size=2)
    print(data_module.vocab_size)

    print("Train dataset size:", len(data_module.train_dataset))
    for batch in data_module.train_dataloader():
        print(batch.shape)
        print(batch)
        break
    
    print("Validation dataset size:", len(data_module.val_dataset))
    for batch in data_module.val_dataloader():
        print(batch.shape)
        break

    print("Test dataset size:", len(data_module.test_dataset))
    for batch in data_module.test_dataloader():
        print(batch.shape)
        break