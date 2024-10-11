

import time

import torch
from gigmate.dataset.dataset import get_data_loader, get_inputs_and_targets, restore_initial_sequence
from gigmate.utils.device import Device, get_device
from torch.utils.data import DataLoader


def iterate_over_data_loader(data_loader: DataLoader, device: Device) -> int:
    total_items = 0

    for batch in data_loader:
        inputs, labels, sequence_lengths = get_inputs_and_targets(batch, device)
        shape = inputs.shape
        total_items += shape[0]

    return total_items


def test_train_data_loader():
    data_loader = get_data_loader('train')
    iterate_over_data_loader(data_loader, get_device())


def test_validation_data_loader():
    data_loader = get_data_loader('validation')
    iterate_over_data_loader(data_loader, get_device())


def test_test_data_loader():
    data_loader = get_data_loader('test')
    iterate_over_data_loader(data_loader, get_device())


def test_restore_initial_sequence():
    data_loader = get_data_loader('validation')
    device = get_device()
    iterator = iter(data_loader)
    batch = next(iterator)
    inputs, targets, sequence_lengths = get_inputs_and_targets(batch, device)
    first_element = inputs[:1, :4, :]
    sequence = restore_initial_sequence(first_element, sequence_lengths[0])

    assert sequence.shape == (1, 4, first_element.shape[2] - 3)
    assert torch.equal(sequence[:, 0:4, 0:1], torch.cat([first_element[:, 0:1, 0:1], first_element[:, 1:2, 1:2], first_element[:, 2:3, 2:3], first_element[:, 3:4, 3:4]], dim=1)), 'Sequences do not match'


def measure_dataloader_iteration_time(data_loader: DataLoader, device: Device):
    print('Measuring access time to dataloader...')
    start_time = time.time()
    total_items = iterate_over_data_loader(data_loader, device)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total time to iterate: {total_time:.2f} seconds")
    print(f"Total number of items: {total_items}")
    print(f"Average time per item: {total_time / total_items:.5f} seconds")
    print(f"Items per second: {total_items / total_time:.2f}")


if __name__ == '__main__':
    data_loader = get_data_loader('train')
    device = get_device()
    measure_dataloader_iteration_time(data_loader, device)