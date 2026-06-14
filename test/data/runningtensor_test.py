import torch
import pytest

from bhtrace.data import RunningTensor

def test_running_tensor_init():
    x0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rt = RunningTensor(x0)
    assert torch.equal(rt.x, x0)
    assert rt.shape == (4,)
    assert not rt.trace
    assert len(rt._diffs) == 0

def test_running_tensor_update():
    x0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rt = RunningTensor(x0, trace=True)

    mask1 = torch.tensor([True, False, True, False])
    update_val1 = torch.tensor([10.0, 30.0])
    rt.update(update_val1, mask1)

    expected_x1 = torch.tensor([10.0, 2.0, 30.0, 4.0])
    assert torch.equal(rt.x, expected_x1)
    assert len(rt._diffs) == 1
    assert torch.equal(rt._diffs[0], update_val1)

    mask2 = torch.tensor([False, True, True, False])
    update_val2 = torch.tensor([20.0, 35.0])
    rt.update(update_val2, mask2)

    expected_x2 = torch.tensor([10.0, 20.0, 35.0, 4.0])
    assert torch.equal(rt.x, expected_x2)
    assert len(rt._diffs) == 2
    assert torch.equal(rt._diffs[1], update_val2)

def test_running_tensor_inflate_single_update():
    x0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rt = RunningTensor(x0, trace=True)

    mask1 = torch.tensor([True, False, True, False])
    update_val1 = torch.tensor([10.0, 30.0])
    rt.update(update_val1, mask1)

    # Inflate should reconstruct the state as if starting from zeros
    # and applying the diff.
    # The non-masked parts remain zero as per the implementation.
    inflated_tensor = rt.inflate([mask1])
    expected_inflated = torch.tensor([10.0, 0.0, 30.0, 0.0])
    assert torch.equal(inflated_tensor, expected_inflated)

def test_running_tensor_inflate_multiple_updates():
    x0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rt = RunningTensor(x0, trace=True)

    mask1 = torch.tensor([True, False, True, False])
    update_val1 = torch.tensor([10.0, 30.0])
    rt.update(update_val1, mask1)

    mask2 = torch.tensor([False, True, True, False])
    update_val2 = torch.tensor([20.0, 35.0])
    rt.update(update_val2, mask2)
    
    # After update, self.x is [10.0, 20.0, 35.0, 4.0]

    # Inflate should reconstruct the state by applying diffs in order
    # starting from zeros.
    inflated_tensor = rt.inflate([mask1, mask2])
    expected_inflated = torch.tensor([10.0, 20.0, 35.0, 0.0])
    assert torch.equal(inflated_tensor, expected_inflated)

def test_running_tensor_inflate_mismatched_masks():
    x0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rt = RunningTensor(x0, trace=True)

    mask1 = torch.tensor([True, False, True, False])
    update_val1 = torch.tensor([10.0, 30.0])
    rt.update(update_val1, mask1)

    with pytest.raises(ValueError, match="The number of provided masks"):
        rt.inflate([]) # Mismatched number of masks

    with pytest.raises(ValueError, match="The number of provided masks"):
        rt.inflate([mask1, mask1]) # Too many masks
