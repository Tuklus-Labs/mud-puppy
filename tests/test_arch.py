"""Tests for mud_puppy.arch.

These run without a GPU: the override hook lets us force any arch. The
hardware probes are exercised indirectly via ``get_arch`` on the real
device when one is present.
"""
from __future__ import annotations

import os
import pytest

from mud_puppy import arch
from mud_puppy.arch import ArchFamily


@pytest.fixture(autouse=True)
def _clear_cache_and_env(monkeypatch):
    """Each test gets a fresh detection cache and no override leaking in."""
    arch.clear_cache()
    monkeypatch.delenv("MUD_PUPPY_ARCH_OVERRIDE", raising=False)
    yield
    arch.clear_cache()


def test_gfx1100_is_rdna3(monkeypatch):
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "gfx1100")
    info = arch.get_arch()
    assert info.family == ArchFamily.RDNA3
    assert info.wavefront_size == 32
    assert info.has_matrix_cores is True
    assert info.has_hw_fp8 is False
    assert info.has_hbm is False
    assert info.is_rdna and not info.is_cdna
    assert info.is_amd


def test_gfx942_is_cdna3(monkeypatch):
    """MI300X target. Drives multi-GPU training on the droplet."""
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "gfx942")
    info = arch.get_arch()
    assert info.family == ArchFamily.CDNA3
    assert info.wavefront_size == 64
    assert info.has_matrix_cores is True
    assert info.has_hw_fp8 is True
    assert info.has_hbm is True
    assert info.is_cdna and not info.is_rdna


def test_gfx90a_is_cdna2(monkeypatch):
    """MI210/MI250X target. Wavefront 64, HBM, no hardware FP8."""
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "gfx90a")
    info = arch.get_arch()
    assert info.family == ArchFamily.CDNA2
    assert info.wavefront_size == 64
    assert info.has_matrix_cores is True
    assert info.has_hw_fp8 is False
    assert info.has_hbm is True


def test_gfx908_is_cdna1(monkeypatch):
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "gfx908")
    info = arch.get_arch()
    assert info.family == ArchFamily.CDNA1
    assert info.wavefront_size == 64
    assert info.has_hbm is True


def test_gfx1200_is_rdna4(monkeypatch):
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "gfx1200")
    info = arch.get_arch()
    assert info.family == ArchFamily.RDNA4
    assert info.wavefront_size == 32
    assert info.has_hw_fp8 is True  # RDNA4 ships hardware FP8
    assert info.has_hbm is False


def test_gfx1030_is_rdna2_no_matrix(monkeypatch):
    """RDNA2 has no WMMA/MFMA; kernels should route around matrix cores."""
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "gfx1030")
    info = arch.get_arch()
    assert info.family == ArchFamily.RDNA2
    assert info.has_matrix_cores is False


def test_gfx1010_is_rdna1_no_matrix(monkeypatch):
    """RDNA1 (5700 XT) is distinct from RDNA2 but also has no matrix cores."""
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "gfx1010")
    info = arch.get_arch()
    assert info.family == ArchFamily.RDNA1
    assert info.wavefront_size == 32
    assert info.has_matrix_cores is False
    assert info.has_hbm is False
    assert info.is_rdna
    assert info.mem_bw_gbs_hint > 0.0


def test_rdna1_override_maps_to_gfx1010(monkeypatch):
    """String-form override for RDNA1 maps to a canonical gfx1010 target."""
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "rdna1")
    info = arch.get_arch()
    assert info.family == ArchFamily.RDNA1
    assert info.gfx_target == "gfx1010"
    assert info.has_matrix_cores is False


def test_gfx90c_is_not_cdna(monkeypatch):
    """Renoir/Cezanne APU (gfx90c) must NOT parse as CDNA2.

    Integer parsing of "gfx90c" yields 90, which previously collided with
    the gfx90a bucket. APUs have 32-wide wavefronts, no MFMA, no HBM.
    """
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "gfx90c")
    info = arch.get_arch()
    assert info.is_cdna is False
    assert info.wavefront_size == 32
    assert info.family == ArchFamily.APU
    assert info.has_matrix_cores is False
    assert info.has_hbm is False
    assert info.has_hw_fp8 is False


def test_unknown_gfx_falls_back(monkeypatch):
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "gfx9999")
    info = arch.get_arch()
    assert info.family == ArchFamily.UNKNOWN


def test_family_name_override(monkeypatch):
    """Overriding with a family string also works."""
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "cdna3")
    info = arch.get_arch()
    assert info.family == ArchFamily.CDNA3
    assert info.gfx_target == "gfx942"


def test_mem_bw_monotonic_for_cdna(monkeypatch):
    """CDNA3 (HBM3) has higher bandwidth than CDNA2 (HBM2e)."""
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "cdna2")
    cdna2 = arch.get_arch()
    arch.clear_cache()
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "cdna3")
    cdna3 = arch.get_arch()
    assert cdna3.mem_bw_gbs_hint > cdna2.mem_bw_gbs_hint


def test_recommended_num_warps_cdna(monkeypatch):
    """With wave=64 a 256-wide tile wants fewer warps than on RDNA3."""
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "cdna3")
    cdna = arch.get_arch()
    arch.clear_cache()
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "rdna3")
    rdna = arch.get_arch()
    # Same tile (BM=128, BN=128) -> fewer warps on CDNA (wave=64)
    assert arch.recommended_num_warps(cdna, 128, 128) <= \
        arch.recommended_num_warps(rdna, 128, 128)


def test_recommended_num_stages_hbm_deeper(monkeypatch):
    """HBM parts want deeper pipelining (higher latency tolerance)."""
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "cdna3")
    cdna = arch.get_arch()
    arch.clear_cache()
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "rdna3")
    rdna = arch.get_arch()
    assert arch.recommended_num_stages(cdna) > arch.recommended_num_stages(rdna)


def test_cache_is_cleared_between_probes(monkeypatch):
    """clear_cache() is required for tests that switch override mid-run."""
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "rdna3")
    a = arch.get_arch()
    # Without clear_cache, second lookup would still return RDNA3.
    arch.clear_cache()
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "cdna3")
    b = arch.get_arch()
    assert a.family != b.family


def test_override_priority_over_real_device():
    """Override env var must beat any real-device detection.

    Matters for the droplet workflow: Gary can set
    ``MUD_PUPPY_ARCH_OVERRIDE=cdna3`` to cross-tune kernels from his
    workstation even though the GPU present is a 7900 XTX.
    """
    os.environ["MUD_PUPPY_ARCH_OVERRIDE"] = "cdna3"
    try:
        info = arch.get_arch()
        assert info.family == ArchFamily.CDNA3
    finally:
        os.environ.pop("MUD_PUPPY_ARCH_OVERRIDE", None)


def test_is_fp8_runnable_requires_hw_fp8(monkeypatch):
    """FP8 runtime probe returns False on RDNA3 even if _scaled_mm exists."""
    monkeypatch.setenv("MUD_PUPPY_ARCH_OVERRIDE", "rdna3")
    assert arch.is_fp8_runnable() is False


def test_unknown_device_is_safe():
    """CPU / no-GPU returns UNKNOWN with zeroed fields; never raises."""
    info = arch.get_arch()  # no override, no GPU expected in CI
    # Either it's a real GPU we're running on or UNKNOWN. Both are valid.
    assert info.family in ArchFamily
    # Zero bandwidth is only legal for UNKNOWN.
    if info.family == ArchFamily.UNKNOWN:
        assert info.mem_bw_gbs_hint == 0.0
        assert info.cu_count == 0
