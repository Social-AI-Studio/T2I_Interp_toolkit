import torch as t
from t2Interp.T2I import T2IModel 
import gc
from tqdm import tqdm
from typing import Iterator, Optional, Union
from dictionary_learning. import NNsightActivationBuffer

# from config import DEBUG

# if DEBUG:
#     tracer_kwargs = {'scan' : True, 'validate' : True}
# else:
#     tracer_kwargs = {'scan' : False, 'validate' : False}


# def iter_activations(
#     path: str,
#     batch_size: int,
#     *,
#     key: Optional[str] = None,          # if the .pt contains a dict, specify the tensor key
#     shuffle: bool = True,
#     drop_last: bool = False,            # if False, the last smaller batch is yielded
#     loop: bool = False,                 # True -> cycle forever
#     dtype: Optional[t.dtype] = None,# cast on load (e.g., torch.float16)
#     device: Union[str, t.device] = "cpu"  # usually keep 'cpu'; your buffer moves to its device
# ) -> Iterator[t.Tensor]:
#     """
#     Yields [B, D] activation batches from a .pt file containing a tensor [N, D]
#     (or a dict with such a tensor at `key`). If loop=False, one pass then StopIteration.
#     """
#     # Load once (on CPU by default)
#     obj = t.load(path, map_location="cpu")
#     if isinstance(obj, dict):
#         if key is None:
#             raise ValueError("`.pt` contained a dict; please specify `key=` for the activations tensor.")
#         acts = obj[key]
#     else:
#         acts = obj

#     if not isinstance(acts, t.Tensor) or acts.ndim != 2:
#         raise ValueError(f"Expected a 2D tensor [N, D], got type={type(acts)} shape={getattr(acts, 'shape', None)}")

#     if dtype is not None:
#         acts = acts.to(dtype)
#     if device != "cpu":
#         # Usually keep on CPU and let ActivationBuffer move to its device,
#         # but you can move here if you prefer:
#         acts = acts.to(device)

#     N = acts.size(0)

#     while True:
#         if shuffle:
#             idx = t.randperm(N)
#         else:
#             idx = t.arange(N)

#         # Iterate in contiguous mini-batches
#         for start in range(0, N, batch_size):
#             end = start + batch_size
#             if end > N and drop_last:
#                 break
#             batch = acts[idx[start:end]]
#             if batch.numel() == 0:
#                 continue
#             yield batch

#         if not loop:
#             return

# class ActivationBuffer:
#     """
#     Implements a buffer of activations. The buffer stores activations from a model,
#     yields them in batches, and refreshes them when the buffer is less than half full.
#     """
#     def __init__(self, 
#                  data, # generator which yields text data
#                  model : T2IModel, # LanguageModel from which to extract activations
#                  submodule, # submodule of the model from which to extract activations
#                  d_submodule=None, # submodule dimension; if None, try to detect automatically
#                 #  io='out', # can be 'in' or 'out'; whether to extract input or output activations
#                  n_ctxs=3e4, # approximate number of contexts to store in the buffer
#                  ctx_len=128, # length of each context
#                  refresh_batch_size=512, # size of batches in which to process the data when adding to buffer
#                  out_batch_size=8192, # size of batches in which to yield activations
#                  device='cpu', # device on which to store the activations
#                  remove_bos: bool = False,
#                  ):
#         self.activations = t.empty(0, d_submodule, device=device, dtype=model.dtype)
#         self.read = t.zeros(0).bool()

#         self.data = data
#         self.model = model
#         self.submodule = submodule
#         self.d_submodule = d_submodule
#         self.n_ctxs = n_ctxs
#         self.ctx_len = ctx_len
#         self.activation_buffer_size = n_ctxs * ctx_len
#         self.refresh_batch_size = refresh_batch_size
#         self.out_batch_size = out_batch_size
#         self.device = device
#         self.remove_bos = remove_bos
    
#     def __iter__(self):
#         return self

#     def __next__(self):
#         """
#         Return a batch of activations
#         """
#         with t.no_grad():
#             # if buffer is less than half full, refresh
#             if (~self.read).sum() < self.activation_buffer_size // 2:
#                 self.refresh()

#             # return a batch
#             unreads = (~self.read).nonzero().squeeze()
#             idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:self.out_batch_size]]
#             self.read[idxs] = True
#             return self.activations[idxs]
    
#     def text_batch(self, batch_size=None):
#         """
#         Return a list of text
#         """
#         if batch_size is None:
#             batch_size = self.refresh_batch_size
#         try:
#             return [
#                 next(self.data) for _ in range(batch_size)
#             ]
#         except StopIteration:
#             raise StopIteration("End of data stream reached")
    
#     def tokenized_batch(self, batch_size=None):
#         """
#         Return a batch of tokenized inputs.
#         """
#         texts = self.text_batch(batch_size=batch_size)
#         return self.model.tokenizer(
#             texts,
#             return_tensors='pt',
#             max_length=self.ctx_len,
#             padding=True,
#             truncation=True
#         )

#     def refresh(self):
#         gc.collect()
#         t.cuda.empty_cache()
#         self.activations = self.activations[~self.read]

#         current_idx = len(self.activations)
#         new_activations = t.empty(self.activation_buffer_size, self.d_submodule, device=self.device, dtype=self.model.dtype)

#         new_activations[: len(self.activations)] = self.activations
#         self.activations = new_activations

#         while current_idx < self.activation_buffer_size:
#             with t.no_grad():
#                 with self.model.trace(
#                     self.text_batch(),
#                     **tracer_kwargs,
#                     invoker_args={"truncation": True, "max_length": self.ctx_len},
#                 ):
#                     hidden_states = self.submodule.value.save()
#                     self.submodule.output.stop()

#             hidden_states = hidden_states.value
#             if isinstance(hidden_states, tuple):
#                 hidden_states = hidden_states[0]

#             remaining_space = self.activation_buffer_size - current_idx
#             assert remaining_space > 0
#             hidden_states = hidden_states[:remaining_space]

#             self.activations[current_idx : current_idx + len(hidden_states)] = hidden_states.to(
#                 self.device
#             )
#             current_idx += len(hidden_states)

#         self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

#     @property
#     def config(self):
#         return {
#             'd_submodule' : self.d_submodule,
#             'io' : self.io,
#             'n_ctxs' : self.n_ctxs,
#             'ctx_len' : self.ctx_len,
#             'refresh_batch_size' : self.refresh_batch_size,
#             'out_batch_size' : self.out_batch_size,
#             'device' : self.device
#         }

#     def close(self):
#         """
#         Close the text stream and the underlying compressed file.
#         """
#         self.text_stream.close()

class t2IActivationBuffer(NNsightActivationBuffer):
    """
    Implements a buffer of activations. The buffer stores activations from a model,
    yields them in batches, and refreshes them when the buffer is less than half full.
    """

    def __init__(
        self,
        data,  # generator which yields text data
        model: T2IModel,  # LanguageModel from which to extract activations
        submodule,  # submodule of the model from which to extract activations
        d_submodule=None,  # submodule dimension; if None, try to detect automatically
        # io="out",  # can be 'in' or 'out'; whether to extract input or output activations, "in_and_out" for transcoders
        n_ctxs=3e4,  # approximate number of contexts to store in the buffer
        ctx_len=128,  # length of each context
        refresh_batch_size=512,  # size of batches in which to process the data when adding to buffer
        out_batch_size=8192,  # size of batches in which to yield activations
        device="cpu",  # device on which to store the activations
    ):
        self.activations = t.empty(0, d_submodule, device=device)
        self.read = t.zeros(0).bool()

        self.data = data
        self.model = model
        self.submodule = submodule
        self.d_submodule = d_submodule
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.n_ctxs * self.ctx_len // 2:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[: self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]


    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts, return_tensors="pt", max_length=self.ctx_len, padding=True, truncation=True
        )

    def token_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            return t.tensor([next(self.data) for _ in range(batch_size)], device=self.device)
        except StopIteration:
            raise StopIteration("End of data stream reached")
        
    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        return self.token_batch(batch_size)

    def _reshaped_activations(self, hidden_states):
        hidden_states = hidden_states.value
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        batch_size, seq_len, d_model = hidden_states.shape
        hidden_states = hidden_states.view(batch_size * seq_len, d_model)
        return hidden_states

    def refresh(self):
        self.activations = self.activations[~self.read]

        while len(self.activations) < self.n_ctxs * self.ctx_len:

            with t.no_grad(), self.model.trace(
                self.token_batch(),
                **tracer_kwargs,
                invoker_args={"truncation": True, "max_length": self.ctx_len},
            ):
                hidden_states = self.submodule.value.save()
                hidden_states = self._reshaped_activations(hidden_states)

            self.activations = t.cat([self.activations, hidden_states.to(self.device)], dim=0)
            self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    @property
    def config(self):
        return {
            "d_submodule": self.d_submodule,
            "n_ctxs": self.n_ctxs,
            "ctx_len": self.ctx_len,
            "refresh_batch_size": self.refresh_batch_size,
            "out_batch_size": self.out_batch_size,
            "device": self.device,
        }

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()