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
"""

from __future__ import annotations

import webdataset as wds

import torch

from t2i_interp.utils.T2I.buffer import ActivationsDataloader, PairedLoader


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
