import torch as t
from t2Interp.T2I import T2IModel 
import gc
from tqdm import tqdm
from typing import Iterator, Optional, Union
from dictionary_learning.buffer import NNsightActivationBuffer, tracer_kwargs

# from config import DEBUG

# if DEBUG:
#     tracer_kwargs = {'scan' : True, 'validate' : True}
# else:
#     tracer_kwargs = {'scan' : False, 'validate' : False}

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
        # ctx_len=77,  # length of each context
        refresh_batch_size=512,  # size of batches in which to process the data when adding to buffer
        out_batch_size=512,  # size of batches in which to yield activations
        data_device="cpu",  # device on which to store the activations
        denoising_step=0,  # steps to trace over
        **kwargs,
    ):
        super().__init__(data=data,model=model,submodule=submodule,d_submodule=d_submodule,n_ctxs=n_ctxs,
                         refresh_batch_size=refresh_batch_size,out_batch_size=out_batch_size,device=data_device)
        self.activations = t.empty(0, d_submodule, device=data_device)
        self.read = t.zeros(0).bool()
        self.data = data
        self.model = model
        self.submodule = submodule
        self.d_submodule = d_submodule
        self.n_ctxs = n_ctxs
        # self.ctx_len = ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = data_device
        self.steps = (denoising_step, denoising_step + 1) if isinstance(denoising_step, int) else denoising_step

    def __iter__(self):
        return self

    # def __next__(self):
    #     """
    #     Return a batch of activations
    #     """
    #     with t.no_grad():
    #         # if buffer is less than half full, refresh
    #         if (~self.read).sum() < self.n_ctxs * self.ctx_len // 2:
    #             self.refresh()

    #         # return a batch
    #         unreads = (~self.read).nonzero().squeeze()
    #         idxs = unreads[t.randperm(len(unreads), device=unreads.device)[: self.out_batch_size]]
    #         self.read[idxs] = True
    #         return self.activations[idxs]


    # def tokenized_batch(self, batch_size=None):
    #     """
    #     Return a batch of tokenized inputs.
    #     """
    #     texts = self.text_batch(batch_size=batch_size)
    #     return self.model.tokenizer(
    #         texts, return_tensors="pt", max_length=self.ctx_len, padding=True, truncation=True
    #     )

    def token_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            data = [next(self.data) for _ in range(batch_size)]
            data = {"prompt":[sample if isinstance(sample, str) else "" for sample in data ]}#,
                    # "image": [sample for sample in data if isinstance(sample, t.Tensor)]}
            data = {k:v for k,v in data.items() if len(v)>0}
            return data
        except StopIteration:
            raise StopIteration("End of data stream reached")
    
    # def token_batch(self, batch_size=None):
    #     if batch_size is None:
    #         batch_size = self.refresh_batch_size
    #     texts = [next(self.data) for _ in range(batch_size)]
        
    #     if getattr(self.model, 'tokenizer', None) is None:
    #         raise ValueError("T2I instance does not have a tokenizer")
        
    #     tokenizer = self.model.tokenizer
    #     enc = tokenizer(
    #         texts,
    #         add_special_tokens=True,
    #         padding=True,
    #         truncation=True,
    #         return_tensors="pt",
    #         return_length=True,
    #     )

    #     input_ids_list = []
    #     # attn_list = []
    #     # was_truncated = []

    #     for ids in enc["input_ids"]:
    #         input_ids_list.append(ids)

    #     # pad to a batch (left- or right-pad matches tokenizer’s padding_side)
    #     batch = tokenizer.pad(
    #         {"input_ids": input_ids_list},
    #         padding=True,
    #         max_length=tokenizer.model_max_length,
    #         return_tensors="pt"
    #     )
    #     # # attention mask: 1 for tokens, 0 for pad
    #     # if "attention_mask" not in batch:
    #     #     batch["attention_mask"] = (batch["input_ids"] != tokenizer.pad_token_id).long()

    #     # return {k: v.to(self.device) for k, v in batch.items()}
    #     return tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

    # def text_batch(self, batch_size=None):
    #     """
    #     Return a list of text
    #     """
    #     return self.token_batch(batch_size)

    def _reshaped_activations(self, hidden_states):
        # hidden_states = hidden_states.value
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = hidden_states.view((hidden_states.shape[0],-1))
        return hidden_states

    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.refresh_batch_size:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[: self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]
        
    def refresh(self):
        self.activations = self.activations[~self.read]

        while len(self.activations) < self.refresh_batch_size:
            inputs = self.token_batch()
            with t.no_grad(), self.model.generate(
                **inputs
            ) as tracer:
                with tracer.iter[self.steps[0]: self.steps[1]]:
                    # _ = self.model.pipeline(**self.token_batch())
                    hidden_states = self.submodule.value.save()
                    hidden_states = self._reshaped_activations(hidden_states)
                    self.activations = t.cat([self.activations, hidden_states.to(self.device)], dim=0)
                    tracer.stop()
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

    # def close(self):
    #     """
    #     Close the text stream and the underlying compressed file.
    #     """
    #     self.text_stream.close()