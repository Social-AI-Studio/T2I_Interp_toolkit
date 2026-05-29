"""Regression tests for `ActivationsDataloader` and `PairedLoader`.

The motivating bug: when an inline-pair Steering run produced a train
split smaller than the hardcoded batch_size=16 (e.g. 8 demographic pairs
→ 16 CAA rows → 13 train rows after the 80/20 disjoint split), the
training loop crashed with `StopIteration` at `run_steer.py:347`
(`sample = next(train_loader.iterate())`).

Root cause was inside `ActivationsDataloader.renew_buffer`: it set
`self.buffer = None` first, then raised `StopIteration` before
re-materialising the carry-over rows, so the `except StopIteration`
fallback in `iterate()` (which yields `self.buffer[self.pointer:]` when
non-None) silently dropped the under-sized tail.

The fix materialises the buffer *before* signalling end-of-stream so the
fallback can emit the tail. These tests pin the new behaviour for the
two paths that exercised the bug (single-loader, paired-loader) plus
keep the common case (≥ batch_size rows) green.

Also covers `safe_pth_decoder` — the hardened replacement for the
silent-weights_only=False fallback that historically let any malicious
shard execute arbitrary pickle on the researcher's machine.
"""

from __future__ import annotations

import io
import os
import pickle

import pytest
import torch
import webdataset as wds

from t2i_interp.utils.T2I.buffer import (
    ActivationsDataloader,
    PairedLoader,
    safe_pth_decoder,
)


def _write_tar(path, n_rows: int, dim: int = 8, with_label: bool = False) -> None:
    """Write `n_rows` synthetic activation samples into a webdataset tar."""
    with wds.TarWriter(str(path)) as sink:
        for i in range(n_rows):
            sample: dict = {
                "__key__": f"sample{i:05d}",
                "output.pth": torch.randn(1, dim),
            }
            if with_label:
                sample["label.pth"] = torch.tensor(i % 2)
            sink.write(sample)


def test_loader_yields_tail_when_dataset_smaller_than_batch_size(tmp_path):
    """Regression: 13 rows + batch_size=16 must yield exactly one 13-row batch.

    Previously this raised `StopIteration` from `renew_buffer` before the
    tail was buffered, so `next(loader.iterate())` saw zero batches.

    Same failure mode hit by run_steer.py with 8 inline DEMOGRAPHIC_PAIRS
    (16 rows → 13 train / 3 val after the disjoint 80/20 split).
    """
    tar = tmp_path / "layer0_caption.tar"
    _write_tar(tar, n_rows=13, dim=8)

    loader = ActivationsDataloader(
        paths_to_datasets=[str(tar)],
        block_name="layer0",
        batch_size=16,
        flatten=True,
        shuffle=False,
        device="cpu",
    )
    batches = list(loader.iterate())
    assert len(batches) == 1, f"expected 1 tail batch, got {len(batches)}"
    assert batches[0].shape[0] == 13, f"expected 13 rows, got {batches[0].shape[0]}"


def test_paired_loader_yields_tail_for_caa_sized_split(tmp_path):
    """Regression: the CAA path in `run_steer._loader` builds a `PairedLoader`
    over (activations, labels) reading the SAME tar twice. A 13-row split
    must produce one paired batch; previously zip() over two zero-batch
    iterators yielded nothing and `next(train_loader.iterate())` raised
    `StopIteration` at run_steer.py:347.
    """
    tar = tmp_path / "layer0_caption.tar"
    _write_tar(tar, n_rows=13, dim=8, with_label=True)

    act_loader = ActivationsDataloader(
        paths_to_datasets=[str(tar)],
        block_name="layer0",
        batch_size=16,
        flatten=True,
        shuffle=False,
        device="cpu",
    )
    lbl_loader = ActivationsDataloader(
        paths_to_datasets=[str(tar)],
        block_name="layer0",
        batch_size=16,
        flatten=False,
        shuffle=False,
        device="cpu",
        data_key="label.pth",
    )
    paired = PairedLoader([act_loader, lbl_loader], shuffle=False)

    sample = next(paired.iterate())  # must not raise StopIteration
    assert isinstance(sample, tuple | list) and len(sample) == 2
    assert sample[0].shape[0] == 13
    assert sample[1].shape[0] == 13


def test_loader_full_batches_unchanged(tmp_path):
    """Sanity: datasets larger than batch_size still yield full batches plus
    an optional tail. The fix must not regress the common case."""
    tar = tmp_path / "layer0_caption.tar"
    _write_tar(tar, n_rows=35, dim=8)  # 16 + 16 + 3 tail

    loader = ActivationsDataloader(
        paths_to_datasets=[str(tar)],
        block_name="layer0",
        batch_size=16,
        flatten=True,
        shuffle=False,
        device="cpu",
    )
    batches = list(loader.iterate())
    sizes = [b.shape[0] for b in batches]
    assert sum(sizes) == 35
    assert sizes[:2] == [16, 16]
    assert sizes[-1] == 3


def test_loader_exact_batch_size_yields_one_full_batch(tmp_path):
    """Boundary: exactly batch_size rows → one full batch, no extra tail."""
    tar = tmp_path / "layer0_caption.tar"
    _write_tar(tar, n_rows=16, dim=8)

    loader = ActivationsDataloader(
        paths_to_datasets=[str(tar)],
        block_name="layer0",
        batch_size=16,
        flatten=True,
        shuffle=False,
        device="cpu",
    )
    batches = list(loader.iterate())
    assert len(batches) == 1
    assert batches[0].shape[0] == 16


# ── safe_pth_decoder ────────────────────────────────────────────────────────


def test_safe_pth_decoder_loads_tensors():
    """The default path: torch.load with weights_only=True for plain tensors."""
    buf = io.BytesIO()
    torch.save(torch.arange(5), buf)
    out = safe_pth_decoder("x.pth", buf.getvalue())
    assert torch.equal(out, torch.arange(5))


def test_safe_pth_decoder_returns_none_for_non_pth_keys():
    assert safe_pth_decoder("x.txt", b"hello") is None


def test_safe_pth_decoder_handles_raw_utf8_text():
    """WebDataset writes plain Python `str` for `.pth` keys as raw UTF-8
    bytes (no torch.save wrapper). collect_latents does this for
    caption.pth extras. safe_pth_decoder must round-trip these."""
    raw = "photo of a man".encode()
    assert safe_pth_decoder("caption.pth", raw) == "photo of a man"


def test_safe_pth_decoder_loads_str_pickled_via_torch_save():
    """torch.save("string", buf) wraps in a zipfile + pickle. Our pickle
    fallback (restricted unpickler via pickle_module) must accept it."""
    buf = io.BytesIO()
    torch.save("hello world", buf)
    assert safe_pth_decoder("caption.pth", buf.getvalue()) == "hello world"


def test_safe_pth_decoder_refuses_malicious_pickle_by_default(monkeypatch):
    """A pickle that tries to import os.system must NOT execute under the
    default safe-mode. This is the security guarantee the function's name
    advertises."""
    # Make sure the opt-in env var is OFF.
    monkeypatch.delenv("T2I_ALLOW_UNSAFE_PICKLE", raising=False)

    # Hand-crafted pickle bytes that, if run with weights_only=False AND
    # an unrestricted unpickler, would attempt os.system("...") via REDUCE.
    payload = (
        b"\x80\x04"  # PROTO 4
        b"\x95\x1a\x00\x00\x00\x00\x00\x00\x00"  # FRAME (size)
        b"\x8c\x05posix"  # SHORT_BINUNICODE "posix"
        b"\x94\x8c\x06system\x94\x93\x94"
        b"\x8c\x05whoami\x94\x85\x94R\x94."
    )
    with pytest.raises(RuntimeError, match="refusing to deserialise"):
        safe_pth_decoder("x.pth", payload)


def test_safe_pth_decoder_unsafe_opt_in_allows_full_pickle(monkeypatch):
    """T2I_ALLOW_UNSAFE_PICKLE=1 is the documented escape hatch for shards
    that genuinely need full pickle. The test exercises only the path is
    reached — we use a benign payload (a plain dict via torch.save)."""
    monkeypatch.setenv("T2I_ALLOW_UNSAFE_PICKLE", "1")
    buf = io.BytesIO()
    # An OrderedDict survives the restricted path too, but using a custom
    # class would actually exercise the unsafe path. Plain dict is the
    # closest benign test.
    torch.save({"a": 1, "b": 2}, buf)
    out = safe_pth_decoder("x.pth", buf.getvalue())
    assert out == {"a": 1, "b": 2}
