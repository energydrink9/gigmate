import torch
import pytest
from unittest.mock import Mock, patch
from gigmate.predict import predict_next_note

@pytest.fixture
def mock_model():
    model = Mock()
    model.return_value = torch.tensor([[[0.1, 0.2, 0.7]]])
    return model

def test_predict_next_note_temperature_zero(mock_model):
    input_sequence = torch.tensor([[[1, 2, 3]]])
    result = predict_next_note(mock_model, input_sequence, temperature=0)
    
    assert isinstance(result, torch.Tensor)
    assert result.item() == 2  # argmax of [0.1, 0.2, 0.7]

@patch('torch.multinomial')
def test_predict_next_note_with_temperature(mock_multinomial, mock_model):
    mock_multinomial.return_value = torch.tensor([1])
    input_sequence = torch.tensor([1, 2, 3])
    result = predict_next_note(mock_model, input_sequence, temperature=0.5)
    
    assert isinstance(result, torch.Tensor)
    assert result.item() == 1  # mocked multinomial return value

from torch import nn
import math

def test_loss():
    loss = nn.CrossEntropyLoss(ignore_index=0)
    logits = torch.tensor([[[0.1, 0.2, 100]]]).transpose(1, 2)
    targets = torch.tensor([[2]])
    loss_value = loss(logits, targets)
    print(loss_value)
    softmax = math.exp(0.7) / (math.exp(0.1) + math.exp(0.2) + math.exp(0.7))
    print('computed: ', -math.log(softmax))
    assert math.isclose(loss_value, -math.log(softmax))