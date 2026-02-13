import copy
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from beartype import beartype
from lightning import LightningDataModule
from pytorch_lightning.utilities.seed import isolate_rng
from torch import FloatTensor, Tensor
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", message=".*does not have many workers.*")


class SyntheticCopyDataset(Dataset):
    @beartype
    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        vocab_size: int,
        lookahead: int,
        datatype: str = "int",
        copymode: str = "linear"
    ):
        super().__init__()
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.lookahead = lookahead 
        self.datatype = datatype
        self.copymode = copymode
        self.data, self.task_params = self.gen_data()

    @beartype
    def __len__(self) -> int:
        return self.n_samples

    @beartype
    def __getitem__(
        self, index
    ) -> Any:  # be careful of the return type, please read lightning doc for best-practices
        # Get the data for the given index
        x = self.data["x"][index]
        y = self.data["y"][index]
        return x, y

    @beartype
    @torch.inference_mode()
    def gen_data(
        self,
    ) -> Any:  # be careful on the return type

        if self.datatype == "int": 
            # HERE WE CAN USE -1 AS PADDING TOKEN AS PADDING IS JUST ON THE OUTPUT
            # IF PADDING AN INPUT (THAT WOULD GO THROUGH AN EMBEDDING LAYER) PADDING TOKEN HAS TO BE GREATER THAN 0
            x = torch.randint(0, self.vocab_size, (self.n_samples, self.seq_len), dtype=torch.long)
            y = torch.full((self.n_samples, self.seq_len), -1, dtype=torch.long)  # Initialize with -1 for ignored positions
            if self.copymode == "linear":
                y[:, self.lookahead:] = x[:, :-self.lookahead]
            elif self.copymode == "sin":
                y[:, self.lookahead:] = np.sin(x[:, :-self.lookahead])
           

        elif self.datatype == "real":
            # Generate random sequences of real numbers
            x = torch.randn(self.n_samples, self.seq_len, self.vocab_size)

            # Create the output tensor with p padding tokens at the beginning
            y = torch.zeros_like(x)

            # Compute the shifted output
            if self.copymode == "linear":
                y[:, self.lookahead:] = x[:, :-self.lookahead]
            elif self.copymode == "sin":
                y[:, self.lookahead:] = np.sin(x[:, :-self.lookahead])
            elif self.copymode == "temporal_product":
                # shape: (batch, time - lookahead, lookahead + 1)
                x_unfolded = x.unfold(dimension=1, size=self.lookahead + 1, step=1)  
                prod = x_unfolded.prod(dim=-1)
                sign = prod.sign()
                # eps = 1e-6
                # safe_root = sign * prod.abs().clamp(min=eps).pow(1.0 / (self.lookahead + 1))
                safe_root = sign * prod.abs().pow(1.0 / (self.lookahead + 1))
                y[:, self.lookahead:] = safe_root
            elif self.copymode.startswith("sin_omega"):
                # give the string 'sin_omega_w' -> sin(w*t)
                w = float(self.copymode.split("_")[-1])
                y[:, self.lookahead:] = np.sin(w*x[:, :-self.lookahead])
            elif self.copymode.startswith("sin_t_omega"):
                # give the string 'sin_t_omega_w' -> sin(w*t)
                w = float(self.copymode.split("_")[-1])
                y = np.sin(w*x)
            elif self.copymode == "state_tracking_product":
                #sign*abs(x_1*x_2*...*x_t)^(1/t)
                prod = x.cumprod(dim=1)
                lengths = torch.ones_like(x).cumsum(dim=1)
                y = prod.sign()*(prod.abs().pow(1.0/(lengths))).float()
            elif self.copymode == "state_tracking_product_nonorm":
                #sign*abs(x_1*x_2*...*x_t)
                # x=x+1
                prod=x.cumprod(dim=1)
                y = prod 
            elif self.copymode.startswith("pow"):
                # give the string 'pow_k' -> x^k
                powk = float(self.copymode.split("_")[-1])
                y[:, self.lookahead:] = x[:, :-self.lookahead].pow(powk)
            elif self.copymode.startswith("pow_t"):
                # give the string 'pow_t_k' -> x^k
                powk = float(self.copymode.split("_")[-1])
                y = x.pow(powk)
            elif self.copymode == "state_tracking_sum":
                lengths = torch.ones_like(x).cumsum(dim=1)
                y = (x.cumsum(dim=1)/lengths).float()
            elif self.copymode == "state_tracking_sum_nonorm":
                y = (x.cumsum(dim=1)).float()
            elif self.copymode == "parity":
                x = x.sign()
                y = x.cumprod(dim=1)

        # Make dicts for data and params 
        data_dict = {"x": x, "y": y}
        params_dict = {
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size,
            "lookahead": self.lookahead,
            "n_samples": self.n_samples,
            }

        return data_dict, params_dict


class SyntheticCopyDataModule(LightningDataModule):
    @beartype
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        num_workers: int = 0,
        test_dataset: Optional[Dataset] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # setup

    @beartype
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=None,
        )

    @beartype
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=None,
        )

    @beartype
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=None,
        )

if __name__ == "__main__":
    # Example usage
    train_dataset = SyntheticCopyDataset(n_samples=1000, seq_len=10, vocab_size=5, lookahead=5)
    val_dataset = SyntheticCopyDataset(n_samples=200, seq_len=10, vocab_size=5, lookahead=5)
    test_dataset = SyntheticCopyDataset(n_samples=200, seq_len=10, vocab_size=5, lookahead=5)

    print(f"Train dataset size: {len(train_dataset)}")
    # convert train example to integers
    print(f"Train dataset example (int): {train_dataset[0][0]}")
    print(f"Train dataset example (int): {train_dataset[0][1]}")

    print(f"Test dataset size: {len(val_dataset)}")
    print(f"Test dataset example (int): {val_dataset[0][0]}")
    print(f"Test dataset example (int): {val_dataset[0][1]}")

    data_module = SyntheticCopyDataModule(train_dataset, val_dataset, batch_size=32, test_dataset=test_dataset)

    for batch in data_module.train_dataloader():
        print(batch[0].shape)
        break  # Just to show one batch

    for batch in data_module.val_dataloader():
        print(batch[0].shape)
        break  # Just to show one batch

    for batch in data_module.test_dataloader():
        print(batch[0].shape)
        break  # Just to show one batch

    # real_train_dataset = SyntheticCopyDataset(
    #     n_samples=1000, seq_len=16, vocab_size=5, lookahead=5, datatype="real", copymode='state_tracking_product_nonorm'
    # )
    real_train_dataset = SyntheticCopyDataset(
        n_samples=1000, seq_len=10, vocab_size=5, lookahead=5, datatype="real", copymode="parity"
    )
    real_val_dataset = SyntheticCopyDataset(
        n_samples=200, seq_len=10, vocab_size=5, lookahead=5, datatype="real", copymode="parity"
    )

    print("Train dataset size (real):", len(real_train_dataset))
    print("Test dataset size (real):", len(real_val_dataset))
    print("Train dataset example (real):", real_train_dataset[0][0])
    print("Train dataset example (real):", real_train_dataset[0][1])
    print("Test dataset example (real):", real_val_dataset[0][0])
    print("Test dataset example (real):", real_val_dataset[0][1])
    data_module_real = SyntheticCopyDataModule(
        real_train_dataset, real_val_dataset, batch_size=32
    )
    for batch in data_module_real.train_dataloader():
        print(batch[0].shape)
        break  # Just to show one batch