from miditok.pytorch_data.datasets import _DatasetABC
import pickle
from pathlib import Path
from torch import LongTensor
import json
import os
import torch
from torch.utils.data import IterableDataset
from miditok.utils.split import split_seq_in_subsequences
import math

from gigmate.constants import get_pad_token_id

ITEMS_PER_FILE = 1024 * 16

def pad(sequence, max_seq_len):
    if len(sequence) < max_seq_len:
        return sequence + [get_pad_token_id()] * (max_seq_len - len(sequence))
    else:
        return sequence

def get_item(sequence):
    return {"input_ids": LongTensor(sequence)}


class DatasetPickle(_DatasetABC, IterableDataset):
    r"""
    Basic ``Dataset`` loading JSON files of tokenized music files.

    When indexed (``dataset[idx]``), a ``DatasetJSON`` will load the
    ``files_paths[idx]`` JSON file and return the token ids, that can be used to train
    generative models.

    **This class is only compatible with tokens saved as a single stream of tokens
    (** ``tokenizer.one_token_stream`` **).** If you plan to use it with token files
    containing multiple token streams, you should first split each track token sequence
    with the :py:func:`miditok.pytorch_data.split_dataset_to_subsequences` method.

    If your dataset contains token sequences with lengths largely varying, you might
    want to first split it into subsequences with the
    :py:func:`miditok.pytorch_data.split_files_for_training` method before loading
    it to avoid losing data.

    :param files_paths: list of paths to files to load.
    :param max_seq_len: maximum sequence length (in num of tokens). (default: ``None``)
    :param bos_token_id: *BOS* token id. (default: ``None``)
    :param eos_token_id: *EOS* token id. (default: ``None``)
    """

    def __init__(
        self,
        directory: str,
        max_seq_len: int,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self._effective_max_seq_len = max_seq_len - sum(
            [1 for tok in [bos_token_id, eos_token_id] if tok is not None]
        )

        self.directory = directory
        self.files_paths = list(sorted(Path(directory).glob(f'**/*.pkl')))

        with open(os.path.join(directory, 'metadata.json')) as file:
            metadata = json.load(file)
            total_files = metadata['total_files']
            self.total_files = total_files
            print(f'Loaded dataset with {total_files} files')

        super().__init__()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        total_files = len(self.files_paths)
        if worker_info is None:  # single-process data loading
            start, end = 0, total_files
        else:  # in a worker process
            per_worker = int(math.ceil(total_files / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, total_files)
        
        acc = []
        
        for file_path in self.files_paths[start:end]:
            with open(file_path, 'rb') as file:
                items = pickle.load(file)
                
                for item in items:
                    if len(item) > self.max_seq_len:
                        sequences = split_seq_in_subsequences(item, 0, self.max_seq_len)
                        for sequence in sequences:
                            yield get_item(pad(sequence, self.max_seq_len))
                    elif len(item) == self.max_seq_len:
                        yield get_item(item)
                    else:
                        concatenated = acc + item
                        sequence_length = len(concatenated)
                        if sequence_length < self.max_seq_len:
                            acc = concatenated
                        else:
                            yield get_item(pad(acc, self.max_seq_len))
                            acc = item

        if len(acc) > 0:
            yield get_item(pad(acc, self.max_seq_len))

    # def __len__(self) -> int:
    #     """
    #     Return the size of the dataset.

    #     :return: number of elements in the dataset.
    #     """
    #     return self.total_files
