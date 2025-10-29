from t2Interp.accessors import ModuleAccessor
from abc import ABC, abstractmethod
from loguru import logger
from typing import Any, Dict, Optional, List, Callable, Union, Generator
from dictionary_learning.utils import hf_dataset_to_generator
from utils.utils import BatchIterator, CachedActivationIterator, gen_images_from_prompts
from utils.metrics import MetricBase
import torch as th
from utils.output import Output 
from t2Interp.intervention import run_intervention, ReplaceIntervention
from utils.buffer import t2IActivationBuffer
from utils.text_image_buffer import TextImageActivationBuffer
from contextlib import nullcontext
from utils.runningstats import TrainUpdate, Update
import numpy as np
from utils.utils import convert_buffer_to_memap, ShardedActivationMemmapDataset,ActivationConfig,normalize_gt_batch
from utils.text_image_buffer import _build_buffer
import os
from pathlib import Path
from t2Interp.T2I import T2IModel
from itertools import tee
from copy import deepcopy

class Steer(ABC):
    @abstractmethod
    def fit(self, model:T2IModel, dataset:dict, accessors:ModuleAccessor,**kwargs) -> None:
        pass
    
    @abstractmethod
    def eval(self, *args,**kwargs) -> None:
        pass
    
    @abstractmethod
    def steer(self, *args,**kwargs) -> Any:
        pass
    
class KSteer(Steer):
    def __init__(self, model:T2IModel=None):
        self.model = model
        
    # def fit(self,dataset, accessor, mapper:th.nn.Module,loss_fn: Optional[Callable] = None,
    #         optimizers: List[th.optim.Optimizer]=None,out:Output=None, 
    #         model:T2IModel=None, **kwargs) -> Generator[Update, None, Output]:
        
    #     if self.model is None and model is not None:
    #         self.model = model
    #     assert self.model is not None, "Model must be provided either at initialization or in fit()"
        
    #     if out is not None:
    #         self.out=out
        
    #     def is_tensor(x) -> bool:
    #         return isinstance(x, th.Tensor)

    #     def is_tuple_of_tensors(x) -> bool:
    #         return isinstance(x, tuple) and all(is_tensor(y) for y in x)
   
    #     def log(msg:str):
    #         update = Update(info=msg)
    #         yield update
                
    #     def cache_path(dataset,split,subset):
    #         base = Path("data") / dataset / accessor.attr_name / split
    #         return str(base / str(subset) if subset is not None else base)
        
    #     log(f"Starting KSteer training on dataset {dataset} with accessor {accessor.attr_name}")
    #     generator_train = hf_dataset_to_generator(dataset,**kwargs) 
    #     gt_train = hf_dataset_to_generator(dataset,**{**kwargs,"dataset_column": kwargs.get("ground_truth_column", "ground_truth"),
    #                                                   "preprocess_fn" : kwargs.get("gt_processing_fn", None)}) 
    #     if kwargs.get("use_val", False):
    #         generator_val = hf_dataset_to_generator(dataset,split=kwargs.get("val_split","validation"),**kwargs)
    #         gt_val = hf_dataset_to_generator(dataset,split=kwargs.get("val_split","validation"),
    #                                          **{**kwargs,"dataset_column": kwargs.get("ground_truth_column", "ground_truth"),
    #                                            "preprocess_fn" : kwargs.get("gt_processing_fn", None)})
    #         buffer_val_gt = BatchIterator(gt_val,kwargs.get("out_batch_size", 1))
            
    #     d_sub = kwargs.pop("d_submodule", kwargs.pop("d_submodule", None))
    #     training_device = kwargs.get("training_device", "cpu")
    #     autocast_dtype = kwargs.get("autocast_dtype", th.float32)
        
    #     use_memmap = kwargs.get("use_memmap", False)
        
    #     if use_memmap and os.path.exists(cache_path(dataset,"train",kwargs.get('subset',None))):
    #         log(f"Using existing memmap at {cache_path(dataset,'train',kwargs.get('subset',None))} for training activations")
    #         buffer_train = ShardedActivationMemmapDataset(cache_path(dataset,"train",kwargs.get('subset',None)),**kwargs)
    #     elif use_memmap:
    #         buffer_train = TextImageActivationBuffer(generator_train, self.model, accessor,d_submodule=d_sub, **kwargs) 
    #         log(f"Creating memmap at {cache_path(dataset,'train',kwargs.get('subset',None))} for training activations")
    #         buffer_train = convert_buffer_to_memap(buffer_train,memmap_dir=cache_path(dataset,"train",kwargs.get('subset',None)), **kwargs)
    #     else:
    #         buffer_train = TextImageActivationBuffer(generator_train, self.model, accessor,d_submodule=d_sub, **kwargs)
        
    #     if use_memmap and os.path.exists(cache_path(dataset,"val",kwargs.get('subset',None))) and kwargs.get("use_val", False):
    #         log(f"Using existing memmap at {cache_path(dataset,'val',kwargs.get('subset',None))} for validation activations")
    #         buffer_val = ShardedActivationMemmapDataset(cache_path(dataset,"val",kwargs.get('subset',None)),**kwargs)
    #     elif use_memmap and kwargs.get("use_val", False):
    #         buffer_val = TextImageActivationBuffer(generator_val, self.model, accessor,d_submodule=d_sub, **kwargs)
    #         log(f"Creating memmap at {cache_path(dataset,'val',kwargs.get('subset',None))} for validation activations")
    #         buffer_val = convert_buffer_to_memap(buffer_val,memmap_dir=cache_path(dataset,"val",kwargs.get('subset',None)),**kwargs)
    #     elif kwargs.get("use_val", False):
    #         buffer_val = TextImageActivationBuffer(generator_val, self.model, accessor,d_submodule=d_sub, **kwargs)
        
    #     use_cache = kwargs.get("cache_activations", False)
    #     if use_cache:
    #         buffer_train = CachedActivationIterator(buffer_train, **kwargs)
    #         if kwargs.get("use_val", False):    
    #             buffer_val = CachedActivationIterator(buffer_val, **kwargs)
            
    #     buffer_train_gt  = BatchIterator(gt_train,kwargs.get("out_batch_size", 1))
         
    #     log_steps = kwargs.get("log_steps", 1) 
    #     steps = kwargs.get("train_steps", 1) 
    #     # autocast_context = th.autocast(device_type=training_device, dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
    #     if optimizers is None:
    #         optimizers = [th.optim.Adam(mapper.parameters(), lr=kwargs.get("lr",1e-5))]
    #     mapper = mapper.to(device=training_device, dtype=autocast_dtype) 
        
    #     self.model.eval()
    #     for p in self.model.parameters():
    #         p.requires_grad_(False)  
        
    #     val_losses = [] 
    #     step = 0
    #     log(f"Beginning training for {steps} steps")
    #     while True:
    #         for act,gt in zip(iter(buffer_train),iter(buffer_train_gt)):
    #             # print(125,gt)
    #             act = act.to(training_device, dtype=autocast_dtype)
    #             if type(gt) is list or type(gt) is tuple:
    #                 all_tensor = all(is_tensor(x) for x in gt)
    #                 all_tuple_tensor = all(is_tuple_of_tensors(x) for x in gt)
    #                 if all_tensor:
    #                     gt = th.stack(gt, dim=0)
    #                     gt = gt.to(training_device)
    #                 elif all_tuple_tensor:
    #                     gt = [th.stack(t, dim=0).to(training_device) for t in zip(*gt)]    
                
    #             mapped = mapper(act) 
    #             if type(mapped) is tuple:
    #                 mapped = list(mapped)
    #             if type(mapped) is list and type(gt) is list:
    #                 loss = 0
    #                 for m,g in zip(mapped,gt):
    #                     loss += loss_fn(m,g.to(training_device))
    #             else:
    #                 print(141,gt,mapped)
    #                 loss = loss_fn(mapped, gt)
    #             loss.backward()
            
    #             for opt in optimizers:
    #                 opt.step()
    #                 opt.zero_grad()

    #             if log_steps and step % log_steps == 0:
    #                 # eval
    #                 val_loss=0
    #                 n_samples=0
    #                 if kwargs.get("use_val", False):
    #                     with th.no_grad():                   
    #                         for val_act,gt_val in zip(iter(buffer_val),iter(buffer_val_gt)):
    #                             val_act = val_act.to(device=training_device, dtype=autocast_dtype)
    #                             # gt_val = gt_val.to(training_device)
    #                             if type(gt_val) is list:
    #                                 # gt_val = th.stack(gt_val, dim=0).to(training_device)
    #                                 if type(gt_val) is list or type(gt_val) is tuple:
    #                                     all_tensor = all(is_tensor(x) for x in gt_val)
    #                                     all_tuple_tensor = all(is_tuple_of_tensors(x) for x in gt_val)
    #                                     if all_tensor:
    #                                         gt_val = th.stack(gt_val, dim=0)
    #                                         gt_val = gt_val.to(training_device)
    #                                     elif all_tuple_tensor:
    #                                         gt_val = [th.stack(t, dim=0).to(training_device) for t in zip(*gt_val)]   
                         
    #                             mapped_val = mapper(val_act)
    #                             if type(mapped_val) is tuple:
    #                                 mapped_val = list(mapped_val)
    #                             if type(mapped_val) is list and type(gt_val) is list:
    #                                 for m_val,g_val in zip(mapped_val,gt_val):
    #                                     val_loss += loss_fn(m_val, g_val.to(training_device))
    #                             else:
    #                                 val_loss += loss_fn(mapped_val, gt_val)
    #                             n_samples += 1
      
    #                         val_loss = val_loss / n_samples
    #                         if val_loss < min(val_losses, default=float('inf')):
    #                             best_mapper = mapper.state_dict()
    #                         val_losses.append(val_loss)
    #                     update = TrainUpdate(step=step, parts={"loss": loss.item(), "val_loss": val_loss.item()})    
    #                 else:
    #                     update = TrainUpdate(step=step, parts={"loss": loss.item()})
    #                 yield update
    #             step += 1    
    #             if step >= steps:
    #                 break
    #         if step >= steps:
    #                 break
            
    #     if kwargs.get("use_val", False):
    #        assert "best_mapper" in locals(), "No best mapper found during validation"
    #        mapper = mapper.load_state_dict(best_mapper) 
    #     log("Finished training KSteer mapper")       
    #     self.classifier = mapper 
    #     if not hasattr(self, "out"):
    #         self.out = Output()
    #     self.out.run_metadata = {**kwargs}
    #     self.out.best_ckpt = best_mapper
    #     return self.out

    def fit(
        self,
        dataset,
        accessor,
        mapper: th.nn.Module,
        loss_fn: Optional[Callable] = None,
        optimizers: Optional[List[th.optim.Optimizer]] = None,
        out: Output | None = None,
        model: T2IModel | None = None,
        **kwargs,
    ) -> Generator[Update, None, Output]:

        # resolve model/out
        if self.model is None and model is not None:
            self.model = model
        assert self.model is not None, "Model must be provided at init or in fit()"
        if out is not None:
            self.out = out
        else:
            self.out = Output()    
            
        cfg = ActivationConfig(
            steps=kwargs.get("train_steps", 1),
            log_steps=kwargs.get("log_steps", 1),
            lr=kwargs.get("lr", 1e-5),
            training_device=kwargs.get("training_device", "cpu"),
            autocast_dtype=kwargs.get("autocast_dtype", th.float32),
            grad_clip_norm=kwargs.get("grad_clip_norm", None),
            # d_submodule=kwargs.get("d_submodule", kwargs.get("d_submodule", None)),
            # use_val=kwargs.get("use_val", False),
            # val_split=kwargs.get("val_split", "validation"),
            # use_memmap=kwargs.get("use_memmap", False),
            # cache_activations=kwargs.get("cache_activations", False),
            # subset=kwargs.get("subset", None),
            # out_batch_size=kwargs.get("out_batch_size", 1),
            data_loader_kwargs=kwargs.get("data_loader_kwargs", {}),
            # buffer_kwargs=kwargs.get("buffer_kwargs", {}),
            # pipe_kwargs=kwargs.get("pipe_kwargs", {}),
        )
        
        # data generators
        # print(250,cfg.data_loader_kwargs)
        gen_train = hf_dataset_to_generator(dataset, **cfg.data_loader_kwargs)
        gt_train = hf_dataset_to_generator(
            dataset,
            **{**cfg.data_loader_kwargs,
               "preprocess_fn" : cfg.data_loader_kwargs.get("gt_processing_fn", None),
               "dataset_column": cfg.data_loader_kwargs.get("ground_truth_column", None)},
        )
        gt_train_iter = BatchIterator(gt_train, cfg.data_loader_kwargs.get('out_batch_size',1))

        if cfg.data_loader_kwargs.get("use_val", False):
            gen_val = hf_dataset_to_generator(dataset, **cfg.data_loader_kwargs)
            gt_val_raw = hf_dataset_to_generator(
                dataset,
                **{**cfg.data_loader_kwargs,
                   "preprocess_fn" : cfg.data_loader_kwargs.get("gt_processing_fn", None),
                   "dataset_column": cfg.data_loader_kwargs.get("ground_truth_column", None)},
            )
            gt_val_iter = BatchIterator(gt_val_raw, cfg.data_loader_kwargs.get('out_batch_size',1))
        
        # buffers
        buf_train = _build_buffer(gen_train, self.model, accessor,dataset,"train", cfg)
        buf_val = None
        if cfg.data_loader_kwargs.get("use_val", False):
            buf_val = _build_buffer(gen_val, self.model, accessor,dataset,cfg.data_loader_kwargs.get('split','val'), cfg)

        # optim
        if optimizers is None:
            optimizers = [th.optim.Adam(mapper.parameters(), lr=cfg.lr)]
        mapper = mapper.to(device=cfg.training_device, dtype=cfg.autocast_dtype)

        # freeze model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
            
        # logging
        yield Update(info=f"Starting KSteer training on dataset={dataset} accessor={accessor.attr_name}")
        yield Update(info=f"""Train steps={cfg.steps}, device={cfg.training_device},
                     dtype={cfg.autocast_dtype},memmap={cfg.data_loader_kwargs.get('use_memmap',False)},
                     cached={cfg.data_loader_kwargs.get('cache_activations',False)}""")

        # best ckpt
        best_mapper_sd = deepcopy(mapper.state_dict())
        best_val = float("inf")
        
        # training loop
        step = 0
        while step < cfg.steps:
            # NOTE: if your iterators are finite, re-initialize per epoch
            train_iter = zip(iter(buf_train), iter(gt_train_iter))
            for act, gt in train_iter:
                act = act.to(cfg.training_device, dtype=cfg.autocast_dtype)
                gt = normalize_gt_batch(gt, cfg.training_device)

                mapped = mapper(act)
                if isinstance(mapped, tuple):
                    mapped = list(mapped)

                if isinstance(mapped, list) and isinstance(gt, list):
                    loss = sum(loss_fn(m, g) for m, g in zip(mapped, gt))
                else:
                    loss = loss_fn(mapped, gt)

                loss.backward()
                if cfg.grad_clip_norm is not None:
                    for opt in optimizers:
                        th.nn.utils.clip_grad_norm_(mapper.parameters(), cfg.grad_clip_norm)

                for opt in optimizers:
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                # logging & validation
                if cfg.log_steps and (step % cfg.log_steps == 0):
                    if cfg.data_loader_kwargs.get("use_val",False) and buf_val is not None:
                        mapper.eval()
                        with th.no_grad():
                            val_loss, n = 0.0, 0
                            for val_act, gt_val in zip(iter(buf_val), iter(gt_val_iter)):
                                val_act = val_act.to(cfg.training_device, dtype=cfg.autocast_dtype)
                                gt_val = normalize_gt_batch(gt_val, cfg.training_device)

                                mapped_val = mapper(val_act)
                                if isinstance(mapped_val, tuple):
                                    mapped_val = list(mapped_val)

                                if isinstance(mapped_val, list) and isinstance(gt_val, list):
                                    for m_val, g_val in zip(mapped_val, gt_val):
                                        val_loss += float(loss_fn(m_val, g_val))
                                else:
                                    val_loss += float(loss_fn(mapped_val, gt_val))
                                n += 1
                            val_loss = val_loss / max(n, 1)

                            # best ckpt
                            if val_loss < best_val:
                                best_val = val_loss
                                best_mapper_sd = deepcopy(mapper.state_dict())
                        mapper.train()
                        yield TrainUpdate(step=step, parts={"loss": float(loss.item()), "val_loss": float(val_loss)})
                    else:
                        yield TrainUpdate(step=step, parts={"loss": float(loss.item())})

                step += 1
                if step >= cfg.steps:
                    break

        # restore best (if val used)
        if cfg.data_loader_kwargs.get("use_val",False):
            mapper.load_state_dict(best_mapper_sd)

        yield Update(info="Finished training KSteer mapper")
        yield Update(info="Evaluating KSteer mapper on validation set")
        val_acc = self.eval(
            dataset,
            accessor,
            mapper,
            model=self.model,
            **kwargs,
        )
        yield Update(info=f"KSteer mapper validation accuracy: {val_acc:.4f}")
        self.classifier = mapper
        self.out.run_metadata = {**kwargs}
        self.out.best_ckpt = best_mapper_sd
        return self.out
    
    @th.no_grad()
    def eval(self,
        dataset,
        accessor,
        mapper: th.nn.Module,
        out: Output | None = None,
        model: T2IModel | None = None,
        **kwargs,):
        
        if self.model is None and model is not None:
            self.model = model
        assert self.model is not None, "Model must be provided at init or in fit()"
        
        data_loader_kwargs=kwargs.get("data_loader_kwargs", {})
        # data generators
        # gen_train = hf_dataset_to_generator(dataset, **data_loader_kwargs)
        # gt_train = hf_dataset_to_generator(
        #     dataset,
        #     **{**data_loader_kwargs,
        #        "preprocess_fn" : data_loader_kwargs.get("gt_processing_fn", None),
        #        "dataset_column": data_loader_kwargs.get("ground_truth_column", None)},
        # )
        cfg = ActivationConfig(
            data_loader_kwargs=kwargs.get("data_loader_kwargs", {}),
        )
        
        gen_val = hf_dataset_to_generator(dataset, **data_loader_kwargs)
        gt_val_raw = hf_dataset_to_generator(
            dataset,
            **{**data_loader_kwargs,
                "preprocess_fn" : data_loader_kwargs.get("gt_processing_fn", None),
                "dataset_column": data_loader_kwargs.get("ground_truth_column", None)},
        )
        gt_val_iter = BatchIterator(gt_val_raw, data_loader_kwargs.get('out_batch_size',1))
        
        buf_val = _build_buffer(gen_val, self.model, accessor,dataset,data_loader_kwargs.get('split','val'), cfg)

        mapper.eval()
        # freeze model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
            
        # logging
        # yield Update(info=f"Evaluating KSteer mapper on dataset={dataset} accessor={accessor.attr_name}")
        
        val_correct=0
        val_total = 0
        for val_act, gt_val in zip(iter(buf_val), iter(gt_val_iter)):
            val_act = val_act.to(kwargs.get("training_device", "cpu"), dtype=kwargs.get("autocast_dtype", th.float32))
            gt_val = normalize_gt_batch(gt_val, kwargs.get("training_device", "cpu"))

            mapped_val = mapper(val_act)
            if isinstance(mapped_val, tuple):
                mapped_val = list(mapped_val)

            if isinstance(mapped_val, list) and isinstance(gt_val, list):
                for m_val, g_val in zip(mapped_val, gt_val):
                    _, predicted = th.max(m_val, 1)
                    val_correct += (predicted == g_val).sum().item()
                    val_total += 1
            else:
                _, predicted = th.max(mapped_val, 1)
                val_correct += (predicted == gt_val).sum().item()
                val_total += 1
        return val_correct / val_total
        
    
    @th.no_grad()
    def predict_proba(self, X: th.Tensor) -> np.ndarray:
        self.classifier.eval()
        X = X.to(self.model.device)
        logits = self.classifier(X)
        probs = th.sigmoid(logits)
        return probs.cpu().numpy()
    
    def steering_loss_uniform(
        self,
        logits: th.Tensor,
        # *,
        # target_idx: List[int] | th.Tensor,
        # avoid_idx: List[int] | th.Tensor,
        ) -> th.Tensor:
        loss = 0
        mean_logits = th.mean(logits, dim=1)

        target_indices = [i for i, logits in enumerate(logits) if logits[:,i].mean() < mean_logits]
        avoid_indices = [i for i, logits in enumerate(logits) if logits[:,i].mean() > mean_logits]
        
        if target_indices:
            target_logits = logits[:, target_indices]
            # Negative because we want to maximize these logits (gradient descent will minimize)
            loss = loss - target_logits.mean()
        
        if avoid_indices:
            avoid_logits = logits[:, avoid_indices]
            # Positive because we want to minimize these logits
            loss = loss + avoid_logits.mean()
            
        return loss    
        
        
    def compute_steering_loss(
        self,
        logits: th.Tensor,
        *,
        target_idx: List[int] | th.Tensor,
        avoid_idx: List[int] | th.Tensor,
        ) -> th.Tensor:
            if not th.is_tensor(target_idx):
                target_idx = th.as_tensor(target_idx, device=logits.device)
            else:
                target_idx = target_idx.to(logits.device)
            if not th.is_tensor(avoid_idx):
                avoid_idx = th.as_tensor(avoid_idx, device=logits.device)
            else:
                avoid_idx = avoid_idx.to(logits.device)

            B, _ = logits.shape
            if avoid_idx.numel() > 0:
                avoid_term = logits[:, avoid_idx].mean(dim=1)
            else:
                avoid_term = th.zeros(B, device=logits.device)
            if target_idx.numel() > 0:
                target_term = logits[:, target_idx].mean(dim=0)
            else:
                target_term = th.zeros(B, device=logits.device)
            return avoid_term - target_term
    
    def steer(
        self,
        acts: Union[np.ndarray, th.Tensor],
        target_idx: List[int]| None = None,
        avoid_idx: List[int] | None = None,
        *,
        alpha: float = 1,
        steer_steps: int = 1,
        step_size_decay: float = 1.0,
        mapper: Optional[str] = None,
        **kwargs,
    ) -> th.Tensor:
        th.set_grad_enabled(True)
        
        if not hasattr(self, "classifier") and mapper is not None:
            self.classifier = mapper
            
        assert hasattr(self, "classifier"), "Classifier not found. Please fit the model or provide a classifier_path."    
        
        if avoid_idx is None:
            avoid_idx = []
        if isinstance(acts, np.ndarray):
            acts_t = th.as_tensor(acts, dtype=th.bfloat16, device=self.classifier.device)
        else:
            acts_t = acts.to(self.classifier.device, dtype=th.bfloat16)

        steered = acts_t.detach().clone()
        for step in range(steer_steps):
            curr = steered.clone().requires_grad_(True)
            logits = self.classifier(curr)
            # loss_vec = self.compute_steering_loss(
            #     logits, target_idx=target_idx, avoid_idx=avoid_idx
            # )
            if type(logits) is list: 
                for logit in logits:
                    loss_vec = self.steering_loss_uniform(
                        logit[:,-1:]
                    )
            else:
                loss_vec = self.steering_loss_uniform(
                    logits[:,-1:]
                )
            loss = loss_vec.mean()
            grads = th.autograd.grad(loss, curr, retain_graph=False)[0]
            current_alpha = alpha * (step_size_decay ** step)
            steered = (curr - current_alpha * grads).detach()
        
        th.set_grad_enabled(False)    
        return steered
    
    # def eval(self, *args,**kwargs) -> None:
    #     pass
    
class CAA(Steer):
    def __init__(self, model):
        self.model = model

    def fit(self, dataset:dict, accessors:ModuleAccessor, **kwargs):
        # dataset must have a positive_prompt and a negative_prompt key
        assert "positive_prompt" in dataset, "dataset must have a positive_prompt key"
        # get positive and negative activations on the accessors
        num_inference_steps = kwargs.get("num_inference_steps", None)
        batch_size = kwargs.get("batch_size", 1)
        device = kwargs.get("device", "cpu")
        seed = kwargs.get("seed", None)
        generate_kwargs = {}
        if "guidance_scale" in kwargs:
            generate_kwargs["guidance_scale"] = kwargs["guidance_scale"]        
        
        # get positive activations   
        pos_activations = {accessor.attr_name:[] for accessor in accessors}  
        gen = hf_dataset_to_generator(dataset,**{"dataset_column": "positive_prompt"})
        for b in BatchIterator(gen,batch_size):          
            with self.model.generate(b, num_inference_steps=num_inference_steps, seed=seed, **generate_kwargs) as tracer:
                for accessor in accessors:
                    act = accessor.value
                    pos_activations[accessor.attr_name].append(act.cpu())
                tracer.stop()
                
        # get negative activations
        neg_activations = {accessor.attr_name:[] for accessor in accessors}
        gen = hf_dataset_to_generator(dataset,**{"dataset_column": "negative_prompt"})
        for b in BatchIterator(gen,batch_size):          
            with self.model.generate(b, num_inference_steps=num_inference_steps, seed=seed, **generate_kwargs) as tracer:
                for accessor in accessors:
                    act = accessor.value
                    neg_activations[accessor.attr_name].append(act.cpu())
                tracer.stop()
                
        # compute mean activations
        directions = {}
        for accessor in accessors:
            pos = pos_activations[accessor.attr_name]
            neg = neg_activations[accessor.attr_name]
            pos = th.cat(pos, dim=0)
            neg = th.cat(neg, dim=0)
            mean_pos = pos.mean(dim=0)
            mean_neg = neg.mean(dim=0)
            direction = mean_pos - mean_neg
            # normalize direction
            direction = direction / direction.norm()
            directions[accessor.attr_name] = direction.to(device)                        
        self.directions = directions
        
    def eval(self, metric:MetricBase, eval_prompts=List[str], **kwargs):
        # metrics={}
        # for attr_name,dir in self.directions.items():
        #     logger.info(f"Evaluating direction for {dir}")
        #     output = run_intervention(model=self.model, prompts=eval_prompts, n_steps = 1, start_step=0, end_step=1, seed=40, interventions = [SteeringIntervention(dir)], **kwargs)
        #     metrics[attr_name]=metric.compute(output)    
        # if kwargs.get("maximize_metric", True):
        #     self.direction= self.directions[max(metrics, key=metrics.get)]    
        # else:    
        #     self.direction= self.directions[min(metrics, key=metrics.get)] 
        pass
    
    def steer(self,**kwargs):
        pass 
    
# class ConceptSearch:
#     def __init__(self, model):
#         self.model = model

#     def search_and_steer(self, dataset:dict, accessors:ModuleAccessor, steering_type:Steer, metric:MetricBase, eval_prompts, **kwargs):
#         steering_type.find_directions(dataset, accessors, **kwargs)
#         steering_type.eval(metric, eval_prompts, **kwargs)  
#         steering_type.steer(**kwargs)

        
        