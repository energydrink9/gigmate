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

from gigmate.constants import get_params

ITEMS_PER_FILE = 1024 * 2

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
        self.files_paths = list(sorted(Path(directory).glob(f'**/*.pkl')))[:1]

        with open(os.path.join(directory, 'metadata')) as file:
            metadata = json.load(file)
            total_files = metadata['total_files']
            self.total_files = total_files
            print(f'Loaded dataset with {total_files} files')

        super().__init__()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            start, end = 0, len(self.files_paths)
        else:  # in a worker process
            per_worker = int(math.ceil(len(self.files_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.files_paths))
        
        for file_path in self.files_paths[start:end]:
            with open(file_path, 'rb') as file:
                items = pickle.load(file)
                for item in items:
                    if (self.max_seq_len < len(items)):
                        sequences = split_seq_in_subsequences(item, 0, self._effective_max_seq_len)
                    else:
                        sequences = [item]
                    for sequence in sequences:
                        token_ids = self._preprocess_token_ids(
                            sequence,
                            self.max_seq_len,
                            self.bos_token_id,
                            self.eos_token_id,
                        )
                        yield {"input_ids": LongTensor(token_ids)}

    # def __len__(self) -> int:
    #     """
    #     Return the size of the dataset.

    #     :return: number of elements in the dataset.
    #     """
    #     return self.total_files
