# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention import spectral


class _FakeCalibration:

    def __init__(self) -> None:
        self.layers = {
            0: SimpleNamespace(
                layer_type="local",
                head_dim=8,
                num_kv_heads=2,
                k_eigenvalues=torch.ones(2, 8),
                v_eigenvalues=torch.ones(2, 8),
                k_d_eff=torch.tensor([1.0, 2.0]),
                v_d_eff=torch.tensor([3.0, 2.0]),
            ),
            1: SimpleNamespace(
                layer_type="local",
                head_dim=8,
                num_kv_heads=2,
                k_eigenvalues=torch.ones(2, 8),
                v_eigenvalues=torch.ones(2, 8),
                k_d_eff=torch.tensor([6.0, 1.0]),
                v_d_eff=torch.tensor([2.0, 1.0]),
            ),
            2: SimpleNamespace(
                layer_type="global",
                head_dim=16,
                num_kv_heads=1,
                k_eigenvalues=torch.ones(1, 16),
                v_eigenvalues=torch.ones(1, 16),
                k_d_eff=torch.tensor([4.0]),
                v_d_eff=torch.tensor([5.0]),
            ),
        }

    def get_layer(self, layer_idx: int):
        return self.layers.get(layer_idx)


@pytest.fixture(autouse=True)
def _restore_spectral_state():
    layer_types = set(spectral._COMPRESSED_LAYER_TYPES)
    yield
    spectral._COMPRESSED_LAYER_TYPES.clear()
    spectral._COMPRESSED_LAYER_TYPES.update(layer_types)
    spectral._LAYER_CODEBOOKS.clear()
    spectral._PACKED_DIMS.clear()
    spectral._ALLOC_DIMS.clear()
    spectral._PACK_MAPS.clear()
    spectral._UNPACK_MAPS.clear()


def test_phase2_default_allocation_uses_per_layer_packed_dims(monkeypatch):
    monkeypatch.delenv("SPECTRAL_SHARED_ALLOC", raising=False)
    spectral._COMPRESSED_LAYER_TYPES.clear()
    spectral._COMPRESSED_LAYER_TYPES.update({"global", "local"})

    spectral._init_phase2_codebooks(
        _FakeCalibration(), b_high=2, b_low=1, device="cpu"
    )

    assert spectral._PACKED_DIMS == {0: 6, 1: 7, 2: 11}
    assert spectral._ALLOC_DIMS == spectral._PACKED_DIMS


def test_phase2_shared_allocation_keeps_true_packed_dims(monkeypatch):
    monkeypatch.setenv("SPECTRAL_SHARED_ALLOC", "1")
    spectral._COMPRESSED_LAYER_TYPES.clear()
    spectral._COMPRESSED_LAYER_TYPES.update({"global", "local"})

    spectral._init_phase2_codebooks(
        _FakeCalibration(), b_high=2, b_low=1, device="cpu"
    )

    assert spectral._PACKED_DIMS == {0: 6, 1: 7, 2: 11}
    assert spectral._ALLOC_DIMS == {0: 7, 1: 7, 2: 14}
