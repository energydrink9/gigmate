from miditok.pytorch_data.datasets import _DatasetABC
import pickle
from pathlib import Path
from torch import LongTensor
import json
import os
from gigmate.processing.steps.tokenize import ITEMS_PER_FILE

class DatasetPickle(_DatasetABC):
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
        self.files_paths = list(Path(directory).glob(f'**/*.pkl'))

        with open(os.path.join(directory, 'metadata')) as file:
            metadata = json.load(file)
            total_files = metadata.total_files

        self.items = [None] * total_files
        self.total_files = total_files

        super().__init__()

    def get_item(self, idx):
        if self.items[idx] == None:
            file_idx = idx // ITEMS_PER_FILE
            with open(self.files_paths[file_idx], 'rb') as file:
                items = pickle.load(file)
                for i, item in enumerate(items):
                    self.items[file_idx + i] = item

        return self.items[idx]

    def __getitem__(self, idx: int) -> dict[str, LongTensor]:
        """
        Load the tokens from the ``idx`` JSON file.

        :param idx: index of the file to load.
        :return: the tokens as a dictionary mapping to the token ids as a tensor.
        """
        item = self.get_item(idx)
        token_ids = item['tokens']
        token_ids = self._preprocess_token_ids(
            token_ids,
            self._effective_max_seq_len,
            self.bos_token_id,
            self.eos_token_id,
        )
        return {"input_ids": LongTensor(token_ids)}

    def __len__(self) -> int:
        """
        Return the size of the dataset.

        :return: number of elements in the dataset.
        """
        return self.total_files
