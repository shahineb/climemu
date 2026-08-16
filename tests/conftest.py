"""Shared test fixtures and configuration for climemu tests."""

import pytest
from huggingface_hub import hf_hub_download


@pytest.fixture
def requires_hf():
    """Skip test if HuggingFace Hub is unreachable."""
    try:
        hf_hub_download("shahineb/climemu", "MPI-ESM1-2-LR/monthly/piControl_climatology.nc")
    except Exception:
        pytest.skip("HuggingFace Hub unreachable")
