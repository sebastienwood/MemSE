import pytest
import torch

from MemSE.nn.utils import zero_diag

BS = 8
C = 3
WH = 32

def test_zero_diag():
    batched_square = torch.ones(BS, WH, WH)
    res = zero_diag(batched_square)
    assert res.sum() == batched_square.sum() - WH * BS

    batched_cov = torch.ones(BS, C, WH, WH, C, WH, WH)
    res = zero_diag(batched_cov)
    assert res.sum() == batched_cov.sum() - C * WH * WH * BS
