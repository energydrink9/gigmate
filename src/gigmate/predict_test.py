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
