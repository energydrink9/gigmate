import random
import torch
import pytest
from unittest.mock import Mock, patch
from gigmate.dataset.dataset import SequenceLengths
from gigmate.domain.prediction import predict_next_token
from gigmate.utils.constants import get_random_seed

SEED = get_random_seed()
random.seed(SEED)
torch.manual_seed(SEED)


@pytest.fixture
def mock_model():
    model = Mock()
    model.return_value = torch.tensor([[
        [
            [0.1, 0.2], [0.2, 0.4], [0.1, 0.2],
        ],
        [
            [0.1, 0.2], [0.2, 0.4], [0.4, 0.2],
        ],
        [
            [0.1, 0.2], [0.2, 0.4], [0.1, 0.2],
        ],
        [
            [0.1, 0.2], [0.2, 0.4], [0.2, 0.4],
        ],        
    ]]), None, None

    return model


def test_predict_next_note_temperature_zero(mock_model):
    input_sequence = torch.tensor([[[1, 2, 3]]])
    full_track_sequence = torch.tensor([[[1, 2, 3]]])
    result, _, _ = predict_next_token(
        mock_model,
        input=input_sequence,
        full_track_sequence=full_track_sequence,
        sequence_lengths=SequenceLengths(full_track=[0], stem=[0]),
        current_token_index=2,
        incremental=False,
        temperature=0,
    )

    assert torch.allclose(result, torch.tensor([1, 0, 1, 1]).reshape((1, 4, 1)))


@patch('torch.multinomial')
def test_predict_next_note_with_temperature(mock_multinomial, mock_model):
    mock_multinomial.return_value = torch.tensor([1])
    input_sequence = torch.tensor([[1, 2, 3]])
    full_track_sequence = torch.tensor([[[1, 2, 3]]])
    result, _, _ = predict_next_token(
        mock_model,
        input=input_sequence,
        full_track_sequence=full_track_sequence,
        sequence_lengths=SequenceLengths(full_track=[0], stem=[0]),
        current_token_index=2,
        incremental=False,
        temperature=0.5,
    )

    assert torch.allclose(result, torch.tensor([1, 1, 1, 1]))
