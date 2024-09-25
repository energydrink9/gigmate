import random
import torch
import pytest
from unittest.mock import Mock, patch
from gigmate.domain.prediction import predict_next_token
from gigmate.utils.constants import get_random_seed
from torch import nn
import math

SEED = get_random_seed()
random.seed(SEED)
torch.manual_seed(SEED)

@pytest.fixture
def mock_model():
    model = Mock()
    model.return_value = torch.tensor([[[0.1, 0.2], [0.2, 0.4], [0.1, 0.2]]])
    return model

def test_predict_next_note_temperature_zero(mock_model):
    input_sequence = torch.tensor([[[1, 2, 3]]])
    result = predict_next_token(mock_model, current_token_index=2, input_sequence=input_sequence, incremental=False, temperature=0)
    
    assert isinstance(result, int)
    assert result == 1  # argmax of [0.1, 0.2]

@patch('torch.multinomial')
def test_predict_next_note_with_temperature(mock_multinomial, mock_model):
    mock_multinomial.return_value = torch.tensor([1])
    input_sequence = torch.tensor([[1, 2, 3]])
    result = predict_next_token(mock_model, current_token_index=2, input_sequence=input_sequence, incremental=False, temperature=0.5)
    
    assert isinstance(result, int)
    assert result == 1  # mocked multinomial return value

def test_remove_forbidden_tokens(mock_model):
    input_sequence = torch.tensor([[[1, 2, 3]]])
    result = predict_next_token(mock_model, current_token_index=2, incremental=False, input_sequence=input_sequence, temperature=0, forbidden_tokens=[1])
    
    assert isinstance(result, int)
    assert result == 0  # argmax of [0.1, 0]

def test_loss():
    loss = nn.CrossEntropyLoss(ignore_index=0)
    logits = torch.tensor([[[0.1, 0.2, 0.7]]]).transpose(1, 2)
    targets = torch.tensor([[2]])
    loss_value = loss(logits, targets).item()
    softmax = math.exp(0.7) / (math.exp(0.1) + math.exp(0.2) + math.exp(0.7))
    log_softmax = -math.log(softmax)
    assert math.isclose(loss_value, log_softmax, rel_tol=1e-05)