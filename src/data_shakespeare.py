"""
Resources:
https://huggingface.co/docs/transformers/tasks/language_modeling
https://github.com/dariush-bahrami/character-tokenizer/tree/master


Two options for data loading:

Given a dataset of sequences of different length {s1, s2, ..., s2}, we have two options for dataloading

1. Simple (preprocess_simple)
    - Convert each sequence to be of length `max_len` via padding or trunction

2. Advanced (preprocess_function & group texts)
    - Combine to sinlge length string s = [s_1, s_2, ..., s_b], then split into chunks of size `max_len`. This is less
    - Less wastefulness from truncation


"""

from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from beartype import beartype
from lightning import LightningDataModule
from pytorch_lightning.utilities.seed import isolate_rng
from torch import FloatTensor, Tensor
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from git import Optional
from transformers import default_data_collator
from charactertokenizer import CharacterTokenizer
import string


class BaseChatTemplate:
    @classmethod
    def format_prompt(cls, prompt: str) -> str:
        raise NotImplementedError

    @classmethod
    def get_sample_prompt(cls):
        raise NotImplementedError

    @classmethod
    def safe_parse(cls, generation: str, eos_token: str) -> Optional[str]:
        raise NotImplementedError

    @classmethod
    def check_answer(cls, answer_pred_unp: str, answer_true_unp: str, eos_token: str):
        try:
            answer_pred = cls.safe_parse(answer_pred_unp, eos_token)
            answer_true = cls.safe_parse(answer_true_unp, eos_token)
            if not (answer_pred and answer_true):
                return False
            return answer_pred == answer_true
        except Exception:
            return False


def group_texts(examples, input_seq_len):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])  # type: ignore
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= input_seq_len:
        total_length = (total_length // input_seq_len) * input_seq_len
    # Split by chunks of input_seq_len.
    result = {
        k: [t[i : i + input_seq_len] for i in range(0, total_length, input_seq_len)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


class ChatTemplateShakespeare(BaseChatTemplate):
    @classmethod
    def format_prompt(cls, prompt: str) -> str:
        return prompt

    @classmethod
    def get_sample_prompt(cls):
        return "\n\n"


def load_shakespeare_data(tokenizer, input_seq_len, test_size=0.2, **kwargs):
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # local_path = os.path.join(script_dir, "tinyshakespeare.txt")
    # dataset = load_dataset("text", data_files={"train": local_path}, split="train")
    dataset = load_dataset(
        "tiny_shakespeare", split="train", trust_remote_code=True
    )  # Stopped working on server
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], add_special_tokens=False),
        remove_columns=["text"],
    )
    dataset = dataset.map(lambda x: group_texts(x, input_seq_len), batched=True)
    dataset = dataset.train_test_split(test_size=test_size)  # type: ignore
    # DEBUG: print first example decoded
    # print(f"First example: \n{tokenizer.decode(dataset['train']['input_ids'][0])}")  # type: ignore
    return dataset


class ShakespeareDataModule(LightningDataModule):
    def __init__(self, tokenizer, input_seq_len, batch_size, num_workers=0, test_size=0.2, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.input_seq_len = input_seq_len
        self.test_size = test_size
        self.prepare_data()

    def prepare_data(self):
        # Load initial dataset with train/test split
        initial_dataset = load_shakespeare_data(
            tokenizer=self.tokenizer,
            input_seq_len=self.input_seq_len,
            test_size=self.test_size,
        )
        
        # Split the test set into equal validation and test sets
        test_val_dataset = initial_dataset["test"].train_test_split(test_size=0.5)
        
        # Create a new dataset dictionary with train, validation, and test
        self.dataset = {
            "train": initial_dataset["train"],
            "validation": test_val_dataset["train"],
            "test": test_val_dataset["test"]
        }


    def train_dataloader(self):
        return DataLoader(self.dataset["train"],
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            shuffle=True,
                            collate_fn=default_data_collator)

    def val_dataloader(self):
        return DataLoader(self.dataset["test"],
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            shuffle=False,
                            collate_fn=default_data_collator)
    
    def test_dataloader(self):
        return DataLoader(self.dataset["validation"],
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            shuffle=False,
                            collate_fn=default_data_collator)


# Usage example:
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from charactertokenizer import CharacterTokenizer
    import string


    # Example usage with gpt 2 tokenizer
    #tokenizer = AutoTokenizer.from_pretrained("gpt2")
    #tokenizer.pad_token = tokenizer.eos_token
    
    # Example usage with character level tokenizer
    chars = string.printable # This is character vocab
    model_max_length = 64
    tokenizer = CharacterTokenizer(chars, model_max_length)

    dataset = load_shakespeare_data(
        tokenizer=tokenizer,
        input_seq_len=64,
    )

    print(dataset)
    print(f"\nDataset sizes:")
    print(f"Train: {len(dataset['train'])} sequences")
    print(f"Test: {len(dataset['test'])} sequences")

    print(f"\nFirst batch:")
    batch = next(iter(dataset["train"]))
    print(batch["input_ids"][:50])
    print(tokenizer.decode(batch["input_ids"][:50]))

    data_module = ShakespeareDataModule(
        tokenizer=tokenizer,
        input_seq_len=512,
        batch_size=32,
    )

    for batch in data_module.train_dataloader():
        print(batch["input_ids"].shape)
        print("shape:", batch["input_ids"][0].shape)
        print(tokenizer.decode(batch["input_ids"][0][:50]))
        break

    for batch in data_module.val_dataloader():
        print(tokenizer.decode(batch["input_ids"][0][:50]))
        break
    
    for batch in data_module.test_dataloader():
        print(tokenizer.decode(batch["input_ids"][0][:50]))
        break

    print(data_module.vocab_size)
