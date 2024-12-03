
import torch

from gigmate.training.training_model import weighted_cross_entropy_loss

LARGE_VALUE = 1e20
FIRST_ITEM = [0.9, 0.2, 0.3]
SECOND_ITEM = [0.1, 0.9, 0.3]
THIRD_ITEM = [0.2, 0.2, 0.6]
FIRST_ITEM_ONLY = [LARGE_VALUE, 0, 0]
SECOND_ITEM_ONLY = [0, LARGE_VALUE, 0]
THIRD_ITEM_ONLY = [0, 0, LARGE_VALUE]


def test_weighted_cross_entropy_loss_should_equal_cross_entropy_loss_when_no_weights():
    logits = torch.tensor([FIRST_ITEM, FIRST_ITEM, THIRD_ITEM, FIRST_ITEM, FIRST_ITEM, SECOND_ITEM])
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    weighted_loss = weighted_cross_entropy_loss(logits, targets, torch.tensor([1, 1, 1, 1, 1, 1]))
    loss = torch.nn.functional.cross_entropy(logits, targets)

    assert torch.allclose(loss, weighted_loss)


def test_weighted_cross_entropy_loss_with_all_weights_zero():
    logits = torch.tensor([FIRST_ITEM, FIRST_ITEM, THIRD_ITEM, FIRST_ITEM, FIRST_ITEM, SECOND_ITEM])
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    weighted_loss = weighted_cross_entropy_loss(logits, targets, torch.tensor([0, 0, 0, 0, 0, 0]))

    assert weighted_loss.item() == 0.


def test_weighted_cross_entropy_loss_compared_to_cross_entropy_loss():
    logits = torch.tensor([FIRST_ITEM_ONLY, FIRST_ITEM_ONLY, THIRD_ITEM_ONLY, FIRST_ITEM_ONLY, SECOND_ITEM_ONLY, SECOND_ITEM_ONLY])
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    weights = torch.tensor([0.2, 0.5, 0.3, 0.9, 0.4, 0.1])
    weighted_loss = weighted_cross_entropy_loss(logits, targets, weights)
    loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')

    assert torch.allclose(weighted_loss, (loss * weights).mean())


def test_weighted_cross_entropy_loss_calculation():
    logits = torch.tensor([FIRST_ITEM_ONLY, FIRST_ITEM_ONLY, THIRD_ITEM_ONLY, FIRST_ITEM_ONLY, SECOND_ITEM_ONLY, SECOND_ITEM_ONLY])
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    weights = torch.tensor([0.2, 0.5, 0.3, 0.9, 0.5, 0.2])
    first_error_weight = 0.5
    second_error_weight = 0.2
    weighted_loss = weighted_cross_entropy_loss(logits, targets, weights)
    no_of_items = targets.shape[-1]

    assert torch.allclose(weighted_loss, torch.tensor([((LARGE_VALUE * first_error_weight) + (LARGE_VALUE * second_error_weight)) / no_of_items]))


def test_weighted_cross_entropy_loss_calculation_when_batched():
    logits = torch.tensor([
        [SECOND_ITEM_ONLY, SECOND_ITEM_ONLY, THIRD_ITEM_ONLY, SECOND_ITEM_ONLY, SECOND_ITEM_ONLY, THIRD_ITEM_ONLY],
        [FIRST_ITEM_ONLY, FIRST_ITEM_ONLY, THIRD_ITEM_ONLY, FIRST_ITEM_ONLY, SECOND_ITEM_ONLY, SECOND_ITEM_ONLY],
    ]).permute(0, 2, 1)
    targets = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]])
    weights = torch.tensor([[0.2, 0.5, 0.3, 0.9, 0.5, 0.2]])

    first_error_first_batch_weight = 0.2
    second_error_first_batch_weight = 0.9
    first_error_second_batch_weight = 0.5
    second_error_second_batch_weight = 0.2
    batch_size = targets.shape[0]
    weighted_loss = weighted_cross_entropy_loss(logits, targets, weights)
    no_of_items = batch_size * targets.shape[-1]

    error = LARGE_VALUE * first_error_first_batch_weight + LARGE_VALUE * second_error_first_batch_weight + LARGE_VALUE * first_error_second_batch_weight + LARGE_VALUE * second_error_second_batch_weight

    assert torch.allclose(weighted_loss, torch.tensor([error / no_of_items]))