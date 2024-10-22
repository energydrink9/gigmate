import torch

from gigmate.utils.constants import get_pad_token_id, get_special_tokens
from gigmate.utils.sequence_utils import remove_special_tokens_from_target_and_logits


def test_remove_special_tokens():
    targets = torch.tensor([
        [
            [1, 2, 3, 4, get_pad_token_id(), 0, get_pad_token_id()],
            [1, 2, 3, 4, get_pad_token_id(), 0, get_pad_token_id()],
            [1, 2, 3, 4, get_pad_token_id(), 0, get_pad_token_id()],
            [1, 2, 3, 4, get_pad_token_id(), 0, get_pad_token_id()],
        ],
    ])
    logits = torch.tensor([
        [
            [1, 2, 3, 4, 4, 1, 1],
            [1, 2, 3, 1, 2, 4, 3],
            [1, 2, 3, 4, 1, 0, 3],
            [1, 2, 3, 4, 0, 0, 4],
        ]
    ])
    targets, logits = remove_special_tokens_from_target_and_logits(targets, logits, get_special_tokens())

    assert targets.shape == logits.shape
    assert targets.shape == (1, 4, 5)