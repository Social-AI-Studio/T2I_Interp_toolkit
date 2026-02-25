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
             return torch.load(io.BytesIO(data))
        except Exception:
             # Try text decoding if pickle fails
             try:
                 return data.decode("utf-8")
             except:
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
                
        self.dataset = wds.WebDataset(shard_paths).decode(safe_pth_decoder)
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
        
        for _ in range(to_retrieve):
            sample = next(self.iter)
            
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
                    latents = latents.reshape((-1, latents.shape[-1]))
            else:
                # Add batch dimension if keeping structure (e.g. for PairedLoader)
                latents = latents.unsqueeze(0)
            
            to_merge.append(latents.to(self.device))
            
            current_rows = latents.shape[0]
            self.one_size = current_rows
            
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
    def __init__(self, loaders):
        self.loaders = loaders
        
    def reset(self):
        """Reset all child loaders."""
        for l in self.loaders:
            if hasattr(l, "reset"):
                l.reset()

    def iterate(self):
        iterators = [l.iterate() for l in self.loaders]
        for batch_items in zip(*iterators):
            yield tuple(batch_items)