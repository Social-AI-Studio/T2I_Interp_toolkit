import io
import os
import pickle
import warnings

import torch
import webdataset as wds


# Opt-in env var to allow torch.load(weights_only=False) for legacy tar shards
# that contain non-tensor pickled objects beyond the safe-builtins fallback
# below. Off by default. Set T2I_ALLOW_UNSAFE_PICKLE=1 only for shards you
# produced yourself, since weights_only=False executes arbitrary pickle code.
_UNSAFE_PICKLE_ENV = "T2I_ALLOW_UNSAFE_PICKLE"

# Python builtins safe to reconstruct without code execution. collect_latents
# routinely writes plain `str` / `int` extras (e.g. `caption.pth`, `label.pth`)
# whose pickle opcodes torch.load(weights_only=True) rejects too aggressively.
_SAFE_PICKLE_CLASSES = frozenset(
    {
        ("builtins", name)
        for name in (
            "str",
            "int",
            "float",
            "bool",
            "complex",
            "list",
            "dict",
            "tuple",
            "set",
            "frozenset",
            "bytes",
            "bytearray",
            "NoneType",
        )
    }
    | {("collections", "OrderedDict")}
)


class _SafeBuiltinsUnpickler(pickle.Unpickler):
    """Pickle Unpickler that allow-lists only plain-data Python builtins.

    Refuses to import or reconstruct anything else, including any class that
    would call `__reduce__` / `__setstate__` paths. Safe to run on untrusted
    bytes as long as no allowed type triggers code execution at construction
    (the listed builtins do not).
    """

    def find_class(self, module, name):
        if (module, name) in _SAFE_PICKLE_CLASSES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"safe_pth_decoder: refusing to import {module}.{name} from pickle"
        )


class _SafePickleModule:
    """Drop-in `pickle_module` for `torch.load` that uses _SafeBuiltinsUnpickler.

    `torch.save` writes a zip archive whose `data.pkl` member is plain pickle;
    `torch.load(..., pickle_module=...)` lets us swap the unpickler used to
    read it. Combined with `weights_only=False` this restores the legacy code
    path for loading non-tensor extras (str / int / list / dict) while still
    refusing to import arbitrary classes.
    """

    Pickler = pickle.Pickler
    Unpickler = _SafeBuiltinsUnpickler

    @staticmethod
    def load(file, **kwargs):
        return _SafeBuiltinsUnpickler(file, **kwargs).load()

    @staticmethod
    def loads(data, **kwargs):
        return _SafeBuiltinsUnpickler(io.BytesIO(data), **kwargs).load()


def safe_pth_decoder(key, data):
    """WebDataset decoder for `.pth` shard entries that refuses arbitrary pickle.

    Tries three escalating paths:
      1. `torch.load(weights_only=True)` for tensors — strictest, default.
      2. A class-restricted unpickler that only allows plain Python builtins,
         used for collect_latents's `caption.pth` / `label.pth` extras (str
         and int). Cannot reconstruct user-defined classes or torch modules.
      3. Full `torch.load(weights_only=False)` — disabled by default. Enable
         per-process with `T2I_ALLOW_UNSAFE_PICKLE=1` ONLY for shards you
         produced yourself or fully trust.

    A historical version silently fell back to `weights_only=False` on any
    failure, which executed arbitrary code from any `.tar` shard on disk.
    """
    extension = os.path.splitext(key)[1]
    if extension != ".pth":
        return None
    # WebDataset's TarWriter writes plain Python `str` values for `.pth` keys
    # as raw UTF-8 bytes (no torch.save / pickle wrapper). collect_latents
    # routinely does this for prompt extras like `caption.pth`. Detect that
    # path by looking for the torch.save zip magic and short-circuit.
    if not data.startswith(b"PK\x03\x04") and not data.startswith(b"\x80"):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data
    try:
        return torch.load(io.BytesIO(data), weights_only=True)
    except Exception as torch_err:
        try:
            return torch.load(
                io.BytesIO(data),
                weights_only=False,
                pickle_module=_SafePickleModule,
            )
        except Exception as safe_err:
            if os.environ.get(_UNSAFE_PICKLE_ENV) == "1":
                warnings.warn(
                    f"safe_pth_decoder: safe paths failed for {key!r} "
                    f"(torch.load: {torch_err!r}; safe-unpickler: {safe_err!r}); "
                    f"falling back to unsafe pickle because {_UNSAFE_PICKLE_ENV}=1.",
                    stacklevel=2,
                )
                return torch.load(io.BytesIO(data), weights_only=False)
            raise RuntimeError(
                f"safe_pth_decoder: refusing to deserialise {key!r} "
                f"(torch.load(weights_only=True) failed: {torch_err!r}; "
                f"restricted-builtins unpickler failed: {safe_err!r}). "
                f"If you trust this shard, re-run with {_UNSAFE_PICKLE_ENV}=1."
            ) from safe_err


class ActivationsDataloader:
    def __init__(
        self,
        paths_to_datasets,
        block_name,
        batch_size,
        data_key="output",
        device="cuda",
        num_in_buffer=50,
        seed=None,
        flatten=False,
        transform=None,
        shuffle=True,
    ):
        self.data_key = data_key
        self.device = device
        self.transform = transform
        self.flatten = flatten
        self.shuffle = shuffle

        shard_paths = []
        for p in paths_to_datasets:
            p = os.fspath(p)
            if p.endswith(".tar"):
                shard_paths.append(p)
            else:
                shard_paths.append(os.path.join(p, f"{block_name}.tar"))

        self.dataset = wds.WebDataset(shard_paths, empty_check=False).decode(safe_pth_decoder)
        self.iter = iter(self.dataset)
        self.buffer = None
        self.pointer = 0
        self.num_in_buffer = num_in_buffer
        self.batch_size = batch_size
        self.one_size = None

        self.seed = seed
        self.generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(seed)
        else:
            self.generator.seed()

    def renew_buffer(self, to_retrieve):
        to_merge = []
        if self.buffer is not None and self.buffer.shape[0] > self.pointer:
            to_merge = [self.buffer[self.pointer :].clone()]
        self.buffer = None

        new_loaded = 0
        for _ in range(to_retrieve):
            try:
                sample = next(self.iter)
                new_loaded += 1
            except StopIteration:
                break

            # Use specified key or fallback
            key = self.data_key
            if key == "output":
                key = "output.pth"
            elif key == "diff":
                key = "diff.pth"

            latents = sample[key]

            if self.transform:
                latents = self.transform(latents)

            # Ensure proper dimensions (handle scalars and unbatched samples)
            if latents.ndim == 0:
                latents = latents.unsqueeze(0)

            # Handle shapes
            if latents.ndim == 5:
                latents = latents.permute((0, 1, 3, 4, 2))

            if self.flatten:
                if latents.ndim > 1:
                    # Check if latents is structured as (num_steps, spatial..., channels)
                    # usually num_steps occurs if there are more than 3 dimensions when 1D seq, or 4 dims when 2D spatial.
                    # e.g., (steps, seq, dim) -> 3 dims. (steps, h, w, dim) -> 4 dims.
                    # Webdatasets from `capture_step_index="all"` will always have the step dim at axis 0
                    if getattr(
                        self, "has_multi_step", latents.shape[0] > 1 if latents.ndim >= 2 else False
                    ):
                        steps = latents.shape[0]
                        dim = latents.shape[-1]
                        latents = latents.reshape((steps, -1, dim))
                    else:
                        latents = latents.reshape((-1, latents.shape[-1]))
            else:
                # Add batch dimension if keeping structure (e.g. for PairedLoader)
                latents = latents.unsqueeze(0)

            to_merge.append(latents.to(self.device))

            current_rows = latents.shape[0]
            self.one_size = current_rows

        # No carry-over and no new data → dataset is exhausted
        if not to_merge:
            raise StopIteration

        # Materialise whatever rows we still have into self.buffer BEFORE
        # signalling end-of-stream. The fallback in `iterate()` (the
        # `except StopIteration` branch) yields `self.buffer[self.pointer:]`
        # only when `self.buffer is not None`. The previous order — raise
        # before catting — silently dropped the under-sized tail and
        # `next(loader.iterate())` raised StopIteration whenever the dataset
        # had fewer rows than batch_size (e.g. a 13-row CAA train split
        # against batch_size=16).
        self.buffer = torch.cat(to_merge, dim=0)
        self.pointer = 0

        # No new samples were loaded (dataset exhausted) and the buffered
        # rows alone are fewer than one batch → signal end-of-stream so
        # iterate() can yield the tail and return rather than looping forever.
        if new_loaded == 0 and to_retrieve > 0 and self.buffer.shape[0] < self.batch_size:
            raise StopIteration

        if self.shuffle:
            N = self.buffer.shape[0]
            shuffled_indices = torch.randperm(N, generator=self.generator)
            self.buffer = self.buffer[shuffled_indices]

    def reset(self):
        """Reset the iterator to the beginning of the dataset."""
        self.iter = iter(self.dataset)
        self.buffer = None
        self.pointer = 0
        self.one_size = None

    def iterate(self):
        while True:
            # Buffer loop: Ensure we have at least one batch
            while self.buffer is None or (self.buffer.shape[0] - self.pointer) < self.batch_size:
                try:
                    # Retrieve enough to maybe get a batch
                    to_retrieve = (
                        self.num_in_buffer if self.buffer is None else self.num_in_buffer // 5
                    )
                    self.renew_buffer(to_retrieve)
                except StopIteration:
                    # End of stream. Yield remaining items if any.
                    if self.buffer is not None and self.pointer < self.buffer.shape[0]:
                        yield self.buffer[self.pointer :]
                        self.pointer = self.buffer.shape[0]
                    return

            # Yield full batch
            batch = self.buffer[self.pointer : self.pointer + self.batch_size]
            self.pointer += self.batch_size
            yield batch


class PairedLoader:
    def __init__(self, loaders, shuffle=False, seed=None):
        self.loaders = loaders
        self.shuffle = shuffle
        self.seed = seed

    def reset(self):
        """Reset all child loaders."""
        for l in self.loaders:
            if hasattr(l, "reset"):
                l.reset()

    def iterate(self):
        # Collect all pairs jointly before any shuffling so that
        # sample i from loader A is always paired with sample i from loader B.
        iterators = [l.iterate() for l in self.loaders]
        pairs = list(zip(*iterators, strict=False))
        if self.shuffle:
            rng = torch.Generator()
            if self.seed is not None:
                rng.manual_seed(self.seed)
            idx = torch.randperm(len(pairs), generator=rng).tolist()
            pairs = [pairs[i] for i in idx]
        for batch_items in pairs:
            yield tuple(batch_items)


class InMemoryPairedLoader:
    """Pre-load all activation pairs into a single GPU tensor for fast batching.

    Reads through ``loaders`` once at construction time, concatenates all
    batches into contiguous tensors, and then serves mini-batches purely via
    tensor indexing — no disk I/O during training.

    Drop-in replacement for :class:`PairedLoader` when the full dataset fits
    in GPU memory (typically a few GB for mid-block activations at 5 k samples).

    Args:
        loaders: Sequence of :class:`ActivationsDataloader` (or any object
            with an ``iterate()`` generator).  All loaders must yield the same
            number of batches so that pairing is preserved.
        batch_size: Mini-batch size served during ``iterate()``.
        shuffle: Shuffle sample order on every call to ``iterate()``.
        seed: Optional RNG seed for reproducible shuffles.
        device: Target device for the pre-loaded tensors (e.g. ``"cuda:0"``).
        dtype: Cast tensors to this dtype after loading (``None`` = keep as-is).
    """

    def __init__(
        self,
        loaders,
        batch_size: int = 16,
        shuffle: bool = False,
        seed: int | None = None,
        device: str = "cuda",
        dtype=None,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        print(f"[InMemoryPairedLoader] Loading {len(loaders)} loader(s) into {device} memory...")
        iterators = [l.iterate() for l in loaders]
        # Materialise all batches in lock-step to maintain pairing
        all_batches: list[list] = [[] for _ in loaders]
        for batch_tuple in zip(*iterators, strict=False):
            for i, b in enumerate(batch_tuple):
                all_batches[i].append(b)

        self.tensors: list[torch.Tensor] = []
        for i, batches in enumerate(all_batches):
            t = torch.cat(batches, dim=0)
            if dtype is not None:
                t = t.to(dtype=dtype)
            t = t.to(device=device)
            self.tensors.append(t)
            print(
                f"  loader[{i}]: {t.shape}  {t.dtype}  {t.element_size() * t.numel() / 1e9:.3f} GB"
            )

        self._N = self.tensors[0].shape[0]
        print(f"[InMemoryPairedLoader] Ready — {self._N} samples total.")

    @classmethod
    def from_tensors(
        cls,
        *tensors: torch.Tensor,
        batch_size: int = 16,
        shuffle: bool = False,
        seed: int | None = None,
        device: str = "cuda",
    ) -> "InMemoryPairedLoader":
        """Construct directly from pre-collected tensors (skips the loader iteration step)."""
        obj = object.__new__(cls)
        obj.batch_size = batch_size
        obj.shuffle = shuffle
        obj.seed = seed
        obj.tensors = [t.to(device=device) for t in tensors]
        obj._N = obj.tensors[0].shape[0]
        for i, t in enumerate(obj.tensors):
            print(
                f"  tensor[{i}]: {t.shape}  {t.dtype}  {t.element_size() * t.numel() / 1e9:.3f} GB"
            )
        print(f"[InMemoryPairedLoader] Ready — {obj._N} samples total.")
        return obj

    def reset(self):
        pass  # Nothing to reset; tensors are permanent

    def iterate(self):
        rng = torch.Generator()
        if self.seed is not None:
            rng.manual_seed(self.seed)
        idx = torch.randperm(self._N, generator=rng) if self.shuffle else torch.arange(self._N)
        for start in range(0, self._N, self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            yield tuple(t[batch_idx] for t in self.tensors)
