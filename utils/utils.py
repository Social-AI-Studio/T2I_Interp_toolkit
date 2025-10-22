from __future__ import annotations
from datasets import load_dataset
import zstandard as zstd
import io
import json
import os
from nnsight import LanguageModel
from pydantic import BaseModel
from enum import Enum
import torch
from nnsight.modeling.diffusion import DiffusionModel
from itertools import islice
from typing import Iterable, Iterator, List, TypeVar, Tuple, Any, Mapping
from torchvision import transforms
import numpy as np
from typing import Iterable, Iterator, List, TypeVar, Optional, Callable

T = TypeVar("T")

# from .trainers.top_k import AutoEncoderTopK
# from .trainers.batch_top_k import BatchTopKSAE
# from .trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
# from .trainers.sampledsae import SampledActivationSAE, HybridSampledTopKSAE
# from .dictionary import (
#     AutoEncoder,
#     GatedAutoEncoder,
#     AutoEncoderNew,
#     JumpReluAutoEncoder,
# )

from typing import Any, Mapping, Sequence
from pathlib import Path

def _to_jsonable(x: Any) -> Any:
    # Fast path for JSON primitives
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")

    # Callables → function name
    if callable(x):
        return getattr(x, "__name__", repr(x))

    # Paths
    if isinstance(x, Path):
        return str(x)

    # Mappings: JSON needs string keys
    if isinstance(x, Mapping):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    # Sequences (incl. sets/tuples) → lists
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]

    # NumPy scalars / arrays
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return {"__ndarray__": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    except Exception:
        pass

    # PyTorch tensors / dtypes / devices
    try:
        import torch as th
        if isinstance(x, th.Tensor):
            return {"__tensor__": True, "shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device)}
        if isinstance(x, (th.dtype, th.device)):
            return str(x)
    except Exception:
        pass

    # Fallback: string repr
    return str(x)


def hf_dataset_to_generator(dataset_name, split="train", streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        for x in iter(dataset):
            yield x["text"]

    return gen()


def zst_to_generator(data_path):
    """
    Load a dataset from a .jsonl.zst file.
    The jsonl entries is assumed to have a 'text' field
    """
    compressed_file = open(data_path, "rb")
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(compressed_file)
    text_stream = io.TextIOWrapper(reader, encoding="utf-8")

    def generator():
        for line in text_stream:
            yield json.loads(line)["text"]

    return generator()


def get_nested_folders(path: str) -> list[str]:
    """
    Recursively get a list of folders that contain an ae.pt file, starting the search from the given path
    """
    folder_names = []

    for root, dirs, files in os.walk(path):
        if "ae.pt" in files:
            folder_names.append(root)

    return folder_names


# def load_dictionary(base_path: str, device: str) -> tuple:
#     ae_path = f"{base_path}/ae.pt"
#     config_path = f"{base_path}/config.json"

#     with open(config_path, "r") as f:
#         config = json.load(f)

#     dict_class = config["trainer"]["dict_class"]

#     if dict_class == "AutoEncoder":
#         dictionary = AutoEncoder.from_pretrained(ae_path, device=device)
#     elif dict_class == "GatedAutoEncoder":
#         dictionary = GatedAutoEncoder.from_pretrained(ae_path, device=device)
#     elif dict_class == "AutoEncoderNew":
#         dictionary = AutoEncoderNew.from_pretrained(ae_path, device=device)
#     elif dict_class == "AutoEncoderTopK":
#         k = config["trainer"]["k"]
#         dictionary = AutoEncoderTopK.from_pretrained(ae_path, k=k, device=device)
#     elif dict_class == "BatchTopKSAE":
#         k = config["trainer"]["k"]
#         dictionary = BatchTopKSAE.from_pretrained(ae_path, k=k, device=device)
#     elif dict_class == "MatryoshkaBatchTopKSAE":
#         k = config["trainer"]["k"]
#         dictionary = MatryoshkaBatchTopKSAE.from_pretrained(ae_path, k=k, device=device)
#     elif dict_class == "JumpReluAutoEncoder":
#         dictionary = JumpReluAutoEncoder.from_pretrained(ae_path, device=device)
#     elif dict_class == "SampledActivationSAE":
#         dictionary = SampledActivationSAE.from_pretrained(ae_path, device=device)
#     elif dict_class == "HybridSampledTopKSAE":
#         # l and k are stored in state dict; from_pretrained handles them
#         dictionary = HybridSampledTopKSAE.from_pretrained(ae_path, device=device)
#     else:
#         raise ValueError(f"Dictionary class {dict_class} not supported")

#     return dictionary, config

def encode_prompt(prompt:str, model:DiffusionModel):  
    prompt_embeds, negative_prompt_embeds = model.pipeline.encode_prompt(prompt, model.device, 1, True, None) #tokens for empty prompt
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    return prompt_embeds

def get_submodule(model: LanguageModel, layer: int):
    """Gets the residual stream submodule"""
    model_name = model._model_key
    name_l = model_name.lower()

    if "pythia" in name_l:
        return model.gpt_neox.layers[layer]
    elif "gemma" in name_l:
        return model.model.layers[layer]
    elif "smollm" in name_l or "smol" in name_l:
        # SmolLM-135M follows a standard transformers decoder stack at model.layers
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")

class FieldModel(BaseModel):
    
    class FieldType(Enum):
        
        string = 'string'
        float = 'float'
        integer = 'integer'
    
    name: str
    
    type: FieldModel.FieldType


class InterventionModel(BaseModel):

    name: str

    fields: List[FieldModel] = []
    
    num_instances: int = 0
    
    instances: List = []    
    
class FunctionModule(torch.nn.Module):
    def __init__(self, func: Callable, **bound_kwargs: Any):
        super().__init__()
        self.func = func
        self.bound_kwargs = bound_kwargs  # stored as plain attrs

    def forward(self, *args, **kwargs):
        # kwargs at call-time override the bound ones
        merged = {**self.bound_kwargs, **kwargs}
        return self.func(*args, **merged)    
    
# def batchify(source: Iterable[T], batch_size: int, *, drop_last: bool = False) -> Iterator[List[T]]:
#     it = iter(source)
#     while True:
#         batch = list(islice(it, batch_size))
#         if not batch:
#             break
#         if len(batch) < batch_size and drop_last:
#             break
#         yield batch



class BatchIterator(Iterator[List[T]]):
    """
    Batches items from a (re-iterable) source. Each __iter__ starts a fresh pass.
    If your source is a one-shot iterator, pass source_factory to recreate it.
    """
    def __init__(
        self,
        source: Iterable[T] | Iterator[T],
        batch_size: int,
        drop_last: bool = False,
        *,
        source_factory: Optional[Callable[[], Iterable[T] | Iterator[T]]] = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._src = source
        self._batch_size = int(batch_size)
        self._drop_last = bool(drop_last)
        self._factory = source_factory
        self._it: Optional[Iterator[T]] = None
        self.reset()  # ready for first pass

    def __iter__(self) -> "BatchIterator[T]":
        self.reset()
        return self

    def __next__(self) -> List[T]:
        it = self._it
        if it is None:
            raise StopIteration
        batch = list(islice(it, self._batch_size))
        if not batch:
            raise StopIteration
        if self._drop_last and len(batch) < self._batch_size:
            # drop the tail batch and signal end
            raise StopIteration
        return batch

    def reset(self) -> None:
        """
        Restart iteration from the beginning.
        - If source is re-iterable, make a fresh iterator over it.
        - If source is one-shot, use source_factory to rebuild it.
        """
        if self._factory is not None:
            self._it = iter(self._factory())
        else:
            # If _src is a one-shot iterator and no factory is provided,
            # reset() cannot rewind it.
            if hasattr(self._src, "reset"):
                self._src.reset() 
            self._it = iter(self._src)


        
def preprocess_image(image, target_size=512):
    """
    Preprocess PIL image for VAE encoder.
    
    Parameters:
    -----------
    image : PIL.Image
        Input image
    target_size : int
        Target size (will resize to target_size x target_size)
    
    Returns:
    --------
    torch.Tensor : Preprocessed image tensor [1, 3, H, W] in range [-1, 1]
    """
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Scale to [-1, 1]
    ])
    return transform(image) #.unsqueeze(0)   


# class ActivationMemmapWriter:
#     def __init__(self, path, N, D, dtype=np.float16):
#         self.path = path
#         self.N, self.D = int(N), int(D)
#         self.dtype = dtype
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         # create/overwrite
#         self.mm = np.memmap(path, dtype=dtype, mode="w+", shape=(self.N, self.D))
#         self.pos = 0  # next write row

#     def append_batch(self, acts: torch.Tensor):
#         """
#         acts: (B, D) float tensor on any device
#         """
#         B, D = acts.shape
#         assert D == self.D, f"D mismatch: got {D}, expected {self.D}"
#         end = self.pos + B
#         if end > self.N:
#             raise RuntimeError(f"Memmap capacity exceeded ({end} > {self.N})")
#         # move to CPU numpy
#         arr = acts.detach().to("cpu").to(torch.float16 if self.dtype == np.float16 else torch.float32).numpy()
#         self.mm[self.pos:end, :] = arr
#         self.pos = end

#     def flush(self):
#         self.mm.flush()

#     def close(self):
#         self.flush()
#         del self.mm

class ShardedActivationMemmapDataset:
    """
    Stateful, batch-yielding iterator over sharded activation memmaps.
    Uses a shuffled index list to avoid re-reading. Not thread-safe.
    """
    def __init__(
        self,
        memmap_dir: str,
        # batch_size: int = 1024,
        # device: Optional[str] = None,      # e.g., "cuda" or "cpu"; None -> leave on CPU
        # dtype: t.dtype = t.float32,        # dtype returned to the caller
        pin_memory: bool = False,          # pin only if device is CUDA and you'll H2D copy later
        shuffle: bool = True,
        keep_open_shard: bool = True,       # keep one shard mapped to reduce reopen overhead
        **kwargs,
    ):
        man_p = os.path.join(memmap_dir, "manifest.json")
        with open(man_p, "r") as f:
            man = json.load(f)

        self.memmap_dir = memmap_dir
        # self.batch_size = int(batch_size)
        # self.return_device = device
        # self.return_dtype = dtype
        self.batch_size = int(kwargs.get("out_batch_size", 1))
        self.device = kwargs.get("data_device", None)
        self.dtype = kwargs.get("autocast_dtype", torch.float32)
        
        self.pin_memory = bool(pin_memory)
        self.shuffle = bool(shuffle)
        self.keep_open_shard = bool(keep_open_shard)

        self.feature_dim = int(man["feature_dim"])
        self.shards_meta: List[Tuple[str,int]] = [
            (os.path.join(memmap_dir, sh["path"]), int(sh["rows_written"]))
            for sh in man["shards"]
            if int(sh["rows_written"]) > 0
        ]
        self.total_rows = int(man["total_rows"])
        self._dtype_np = np.dtype(man["dtype"])       # storage dtype
        self._prefix = np.cumsum([0] + [n for _, n in self.shards_meta])  # row offsets

        # iteration state
        self._order = np.arange(self.total_rows, dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng()
            rng.shuffle(self._order)
        self._cursor = 0

        # keep-one-open shard state
        self._open_shard_id = None
        self._open_mm = None

    # -------------------- public API --------------------
    def __len__(self):
        return self.total_rows

    def __iter__(self):
        # reset cursor; reshuffle for a fresh pass
        self._cursor = 0
        if self.shuffle:
            rng = np.random.default_rng()
            rng.shuffle(self._order)
        return self

    def __next__(self) -> torch.Tensor:
        if self._cursor >= self.total_rows:
            # cleanup any open handle
            self._close_open_shard()
            raise StopIteration

        end = min(self._cursor + self.batch_size, self.total_rows)
        idxs = self._order[self._cursor:end]
        self._cursor = end

        batch = self._load_indices(idxs)  # torch float32 by default (configurable)
        # optional device move
        if self.device is not None:
            batch = batch.to(self.device, non_blocking=self.pin_memory)
        return batch

    def reset(self):
        """Manually reset iteration (reshuffles if shuffle=True)."""
        self.__iter__()

    # -------------------- internals --------------------
    def _close_open_shard(self):
        if self._open_mm is not None:
            # explicitly drop ref; memmap closes when GC'd
            del self._open_mm
            self._open_mm = None
            self._open_shard_id = None

    def _open_shard(self, shard_id: int):
        """Open (or reuse) the given shard memmap for reading."""
        if self.keep_open_shard and self._open_shard_id == shard_id and self._open_mm is not None:
            return self._open_mm

        # open fresh
        path, valid = self.shards_meta[shard_id]
        mm = np.memmap(path, dtype=self._dtype_np, mode="r", shape=(valid, self.feature_dim))

        if self.keep_open_shard:
            self._close_open_shard()
            self._open_shard_id = shard_id
            self._open_mm = mm
        return mm

    def _row_to_shard(self, global_idx: int) -> Tuple[int, int]:
        """Map a global row index -> (shard_id, row_in_shard)."""
        # prefix is [0, n0, n0+n1, ...]; find rightmost prefix <= idx
        s = int(np.searchsorted(self._prefix, global_idx, side="right") - 1)
        row = int(global_idx - self._prefix[s])
        return s, row

    def _load_indices(self, idxs: np.ndarray) -> t.Tensor:
        """
        Load a set of global row indices, grouping by shard to minimize opens.
        Returns a torch tensor of shape [B, D] on CPU (optionally pinned),
        with final dtype = self.dtype.
        """
        # group indices by shard
        by_shard = {}
        for g in idxs:
            s, r = self._row_to_shard(int(g))
            by_shard.setdefault(s, []).append(r)

        parts = []
        for s, rows in by_shard.items():
            mm = self._open_shard(s)
            rows = np.asarray(rows, dtype=np.int64)
            # fancy index into memmap -> NumPy array (copy)
            arr = mm[rows, :]                             # shape [k, D], dtype storage
            # to torch
            tens = torch.from_numpy(arr.copy())               # avoid view aliasing on memmap
            tens = tens.to(self.dtype)             # cast to requested dtype
            if self.pin_memory:
                tens = tens.pin_memory()
            parts.append(tens)

        batch = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        return batch

    
def convert_buffer_to_memap(
    buffer,                      # your TextImageActivationBuffer-like object
    memmap_dir: str = "./data",             # directory to put shards + manifest
    D: Optional[int] = None,     # feature dimension; if None inferred from first batch
    shard_rows: int = 100,     # rows per shard (tune to your RAM/IO)
    dtype=np.float16             # stored dtype,
    , **kwargs                   # passed to ShardedActivationMemmapDataset
):
    """
    Stream all unread activations from `buffer` into sharded numpy.memmap files on disk.
    Each shard is shape (shard_rows, D). The last shard may be partially filled; the
    manifest records how many rows are valid.

    The buffer's `__next__()` must return a 2D tensor [B, D] (or [B, ...] which we flatten].
    """
    os.makedirs(memmap_dir, exist_ok=True)

    # Helper to open a new shard memmap
    def _open_shard(shard_id: int, rows: int, D: int):
        path = os.path.join(memmap_dir, f"acts_shard_{shard_id:05d}.memmap")
        mm = np.memmap(path, dtype=dtype, mode="w+", shape=(rows, D))
        return path, mm

    total_rows = 0
    shard_id   = 0
    shard_pos  = 0
    shard_path = None
    shard_mm   = None
    inferred_D = D
    manifest = {
        "dtype": str(np.dtype(dtype)),
        "shard_rows": shard_rows,
        "shards": [],   # list of {"path": str, "rows_written": int, "shape": [rows, D]}
        "total_rows": 0,
        "feature_dim": None,
    }

    # open first shard lazily after first batch (so we can infer D)
    def _ensure_shard():
        nonlocal shard_id, shard_pos, shard_path, shard_mm, inferred_D
        if shard_mm is None:
            shard_path, shard_mm = _open_shard(shard_id, shard_rows, inferred_D)
            shard_pos = 0

    # write a (CPU numpy) batch into current shard, rolling to next as needed
    def _write_batch(arr_np: np.ndarray):
        nonlocal shard_id, shard_pos, shard_mm, shard_path, total_rows
        start = 0
        N = arr_np.shape[0]
        while start < N:
            _ensure_shard()
            capacity = shard_rows - shard_pos
            take = min(capacity, N - start)
            shard_mm[shard_pos:shard_pos+take, :] = arr_np[start:start+take]
            shard_pos += take
            total_rows += take
            start += take
            # if shard filled, flush & record and open a new one on next write
            if shard_pos >= shard_rows:
                shard_mm.flush()
                manifest["shards"].append({
                    "path": os.path.basename(shard_path),
                    "rows_written": shard_rows,
                    "shape": [shard_rows, inferred_D],
                })
                shard_id += 1
                shard_mm = None
                shard_path = None
                shard_pos = 0

    # Iterate until buffer is exhausted
    while True:
        try:
            batch = next(buffer)              # expects [B, D] (or [B, ...])
        except StopIteration:
            break

        if not isinstance(batch, torch.Tensor):
            # support dicts like {"features": tensor} or {"activations": tensor}
            if isinstance(batch, dict):
                # pick first tensor-like item
                for v in batch.values():
                    if isinstance(v, torch.Tensor):
                        batch = v
                        break
            else:
                raise TypeError("Expected tensor or dict with a tensor value from buffer.__next__().")

        # Flatten to [B, D] if needed
        if batch.dim() > 2:
            batch = batch.flatten(1)
        elif batch.dim() == 1:
            batch = batch.unsqueeze(0)

        B, Dnow = batch.shape
        if B == 0:
            continue

        if inferred_D is None:
            inferred_D = int(Dnow)

        # Sanity: check consistent feature dim
        if Dnow != inferred_D:
            raise ValueError(f"Inconsistent feature dim: got {Dnow}, expected {inferred_D}")

        # Move to CPU fp16 numpy
        batch_np = batch.detach().to("cpu")
        if dtype == np.float16:
            batch_np = batch_np.to(torch.float16)
        elif dtype == np.float32:
            batch_np = batch_np.to(torch.float32)
        else:
            # for other dtypes, convert via float32 then astype
            batch_np = batch_np.to(torch.float32)

        batch_np = batch_np.numpy().astype(dtype, copy=False)

        # Write
        _write_batch(batch_np)

        # Optional: free GPU memory sooner
        del batch, batch_np
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Flush last (possibly partial) shard
    if shard_mm is not None:
        shard_mm.flush()
        manifest["shards"].append({
            "path": os.path.basename(shard_path),
            "rows_written": shard_pos,                  # may be < shard_rows
            "shape": [shard_rows, inferred_D],          # file shape; only first rows_written valid
        })

    manifest["total_rows"] = total_rows
    manifest["feature_dim"] = inferred_D

    # Save manifest
    with open(os.path.join(memmap_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    memap_buffer = ShardedActivationMemmapDataset(memmap_dir,**kwargs)
    return memap_buffer

class CachedActivationIterator:
    """
    Consume an iterator of activations ((D,) or (B,D) tensors),
    cache them on CPU (single torch.cat), then iterate batches.

    Usage:
        it = CachedActivationIterator(src_iter, out_batch_size=1024, gpu_device="cuda")
        for xb in it:            # xb is on cuda if gpu_device is set
            ...
        it.reset()               # iterate again
        for xb in it:
            ...
        buf = it.buffer          # CPU cache (N, D), dtype=cpu_dtype
    """
    def __init__(self, it: Iterable[torch.Tensor], **kwargs) -> None:
        # config (all via kwargs)
        self.out_batch_size: int = kwargs.get("out_batch_size", 1024)
        self.cpu_dtype: torch.dtype = kwargs.get("cpu_dtype", torch.float16)
        self.cpu_device: str = kwargs.get("cpu_device", "cpu")
        self.gpu_device: Optional[str] = kwargs.get("gpu_device", None)  # e.g., "cuda" or None to keep on CPU
        self.pin_memory: bool = kwargs.get("pin_memory", True)
        self.non_blocking: bool = kwargs.get("non_blocking", True)

        # build cache immediately
        self._build_cache(it)
        self._i = 0  # iteration cursor

    # -------- public API --------
    def __iter__(self) -> "CachedActivationIterator":
        self._i = 0
        return self

    def __next__(self) -> torch.Tensor:
        if self._i >= self.N:
            raise StopIteration
        j = min(self._i + self.out_batch_size, self.N)
        batch = self.buffer[self._i:j]
        if self.pin_memory:
            batch = batch.pin_memory()
        if self.gpu_device is not None:
            batch = batch.to(self.gpu_device, non_blocking=self.non_blocking)
        self._i = j
        return batch

    def reset(self) -> None:
        """Reset internal cursor so you can iterate again."""
        self._i = 0

    def __len__(self) -> int:
        """Number of items (rows) in the cache."""
        return self.N

    # Expose the CPU cache in case you want random access
    @property
    def buffer(self) -> torch.Tensor:
        return self._buffer

    @property
    def shape(self):
        return tuple(self._buffer.shape)

    # -------- internals --------
    def _build_cache(self, it: Iterable[torch.Tensor]) -> None:
        chunks: List[torch.Tensor] = []
        D: Optional[int] = None

        for x in it:
            x = x.detach()
            if x.dim() == 1:
                x = x.unsqueeze(0)  # (D,) -> (1,D)
            if x.dim() != 2:
                raise ValueError(f"Expected (*, D), got {tuple(x.shape)}")
            if D is None:
                D = x.size(1)
            elif x.size(1) != D:
                raise ValueError(f"Inconsistent feature dim: {x.size(1)} vs expected {D}")
            # keep intermediate on CPU; avoid extra copies
            chunks.append(x.to("cpu", copy=False))

        if not chunks:
            # empty cache
            self._buffer = torch.empty((0, 0), dtype=self.cpu_dtype, device=self.cpu_device)
            self.N = 0
            self.D = 0
            return

        cpu_cat = torch.cat(chunks, dim=0)                # (N, D) CPU
        self._buffer = cpu_cat.to(self.cpu_device, dtype=self.cpu_dtype)
        self.N, self.D = self._buffer.size(0), self._buffer.size(1)
        self._i = 0



