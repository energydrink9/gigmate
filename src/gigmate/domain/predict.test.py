import torch
import pytest
from unittest.mock import Mock, patch
from gigmate.domain.prediction import predict_next_token

@pytest.fixture
def mock_model():
    model = Mock()
    model.return_value = torch.tensor([[[0.1, 0.2, 0.7]]])
    return model

def test_predict_next_note_temperature_zero(mock_model):
    input_sequence = torch.tensor([[[1, 2, 3]]])
    result = predict_next_token(mock_model, input_sequence, temperature=0)
    
    assert isinstance(result, torch.Tensor)
    assert result.item() == 2  # argmax of [0.1, 0.2, 0.7]

@patch('torch.multinomial')
def test_predict_next_note_with_temperature(mock_multinomial, mock_model):
    mock_multinomial.return_value = torch.tensor([1])
    input_sequence = torch.tensor([1, 2, 3])
    result = predict_next_token(mock_model, input_sequence, temperature=0.5)
    
    assert isinstance(result, torch.Tensor)
    assert result.item() == 1  # mocked multinomial return value

from torch import nn
import math

def test_loss():
    loss = nn.CrossEntropyLoss(ignore_index=0)
    logits = torch.tensor([[[0.1, 0.2, 0.7]]]).transpose(1, 2)
    targets = torch.tensor([[2]])
    loss_value = loss(logits, targets).item()
    softmax = math.exp(0.7) / (math.exp(0.1) + math.exp(0.2) + math.exp(0.7))
    log_softmax = -math.log(softmax)
    print(loss_value)
    print('computed: ', log_softmax)
    assert math.isclose(loss_value, log_softmax, rel_tol=1e-05)

def test_remove_forbidden_tokens(mock_model):
    input_sequence = torch.tensor([[[1, 2, 3]]])
    result = predict_next_token(mock_model, input_sequence, temperature=0, forbidden_tokens=[1, 2])
    
    assert isinstance(result, torch.Tensor)
    assert result.item() == 0  # argmax of [0.1, 0, 0]