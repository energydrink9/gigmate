import torch
from gigmate.domain.sampling import sample_from_logits
from gigmate.utils.constants import get_random_seed


VALUE_0 = [1.0, 0.0, 0.0]
VALUE_1 = [0.0, 1.0, 0.0]
VALUE_2 = [0.0, 0.0, 1.0]

LOGITS = [
    [
        [VALUE_1, VALUE_0, VALUE_0],
        [VALUE_1, VALUE_0, VALUE_0],
    ],
    [
        [VALUE_1, VALUE_2, VALUE_1],
        [VALUE_2, VALUE_0, VALUE_1],
    ],
]


def test_sample_from_logits_zero_temperature():
    sample = sample_from_logits(torch.tensor(LOGITS), 0.0)
    
    expected_value = torch.tensor([
        [
            [1, 0, 0],
            [1, 0, 0],
        ],
        [
            [1, 2, 1],
            [2, 0, 1],
        ],
    ])
    
    assert torch.equal(sample, expected_value)
    

def test_sample_from_logits_temperature_one():
    torch.manual_seed(get_random_seed())
    sample = sample_from_logits(torch.tensor(LOGITS), 1.0)
    
    expected_value = torch.tensor([
        [
            [1, 0, 2],
            [1, 2, 1],
        ],
        [
            [2, 2, 1],
            [2, 0, 1],
        ],
    ])
    
    assert torch.equal(sample, expected_value)
    