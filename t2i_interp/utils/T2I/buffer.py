import webdataset as wds
import os
import torch
import numpy as np
import io

# Custom decoder to handle .pth files that might contain unpickled data (e.g. raw strings)
def safe_pth_decoder(key, data):
    extension = os.path.splitext(key)[1]
    if extension == ".pth":
        try:
            # weights_only=True uses a fast, restricted deserializer (tensors only).
            # Falls back to full pickle if the file contains non-tensor objects.
            return torch.load(io.BytesIO(data), weights_only=True)
        except Exception:
            try:
                return torch.load(io.BytesIO(data), weights_only=False)
            except Exception:
                try:
                    return data.decode("utf-8")
                except Exception:
                    return data
    return None

class ActivationsDataloader:
    def __init__(self, paths_to_datasets, block_name, batch_size, data_key='output', device='cuda', num_in_buffer=50, seed=None, flatten=False, transform=None, shuffle=True):
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
        self.generator = torch.Generator(device='cpu')
        if seed is not None:
             self.generator.manual_seed(seed)
        else:
             self.generator.seed()

    def renew_buffer(self, to_retrieve):
        to_merge = []
        if self.buffer is not None and self.buffer.shape[0] > self.pointer:
            to_merge = [self.buffer[self.pointer:].clone()]
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
            if key == 'output': key = 'output.pth'
            elif key == 'diff': key = 'diff.pth'
            
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
                    if getattr(self, "has_multi_step", latents.shape[0] > 1 if latents.ndim >= 2 else False):
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

        # No new samples were loaded (dataset exhausted) and carry-over rows
        # alone are fewer than one batch → signal end-of-stream so iterate()
        # can yield the tail and return rather than looping forever.
        carry_rows = to_merge[0].shape[0] if to_merge else 0
        if new_loaded == 0 and to_retrieve > 0 and carry_rows < self.batch_size:
            raise StopIteration

        self.buffer = torch.cat(to_merge, dim=0)
        
        if self.shuffle:
            N = self.buffer.shape[0]
            shuffled_indices = torch.randperm(N, generator=self.generator)
            self.buffer = self.buffer[shuffled_indices]
        self.pointer = 0



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
                    to_retrieve = self.num_in_buffer if self.buffer is None else self.num_in_buffer // 5
                    self.renew_buffer(to_retrieve)
                except StopIteration:
                    # End of stream. Yield remaining items if any.
                    if self.buffer is not None and self.pointer < self.buffer.shape[0]:
                        yield self.buffer[self.pointer:]
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
        pairs = list(zip(*iterators))
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
        for batch_tuple in zip(*iterators):
            for i, b in enumerate(batch_tuple):
                all_batches[i].append(b)

        self.tensors: list[torch.Tensor] = []
        for i, batches in enumerate(all_batches):
            t = torch.cat(batches, dim=0)
            if dtype is not None:
                t = t.to(dtype=dtype)
            t = t.to(device=device)
            self.tensors.append(t)
            print(f"  loader[{i}]: {t.shape}  {t.dtype}  {t.element_size() * t.numel() / 1e9:.3f} GB")

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
            print(f"  tensor[{i}]: {t.shape}  {t.dtype}  {t.element_size() * t.numel() / 1e9:.3f} GB")
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