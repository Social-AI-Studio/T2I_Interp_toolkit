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
from utils.utils import convert_buffer_to_memap, ShardedActivationMemmapDataset
import os
from pathlib import Path
from t2Interp.T2I import T2IModel
from itertools import tee

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
        
    def fit(self,dataset, accessor, mapper:th.nn.Module,loss_fn: Optional[Callable] = None,
            optimizers: List[th.optim.Optimizer]=None,out:Output=None, 
            model:T2IModel=None, **kwargs) -> Generator[Update, None, Output]:
        
        if self.model is None and model is not None:
            self.model = model
        assert self.model is not None, "Model must be provided either at initialization or in fit()"
        
        if out is not None:
            self.out=out
            
        def log(msg:str):
            update = Update(info=msg)
            yield update
                
        def cache_path(dataset,split,subset):
            base = Path("data") / dataset / accessor.attr_name / split
            return str(base / str(subset) if subset is not None else base)
        
        log(f"Starting KSteer training on dataset {dataset} with accessor {accessor.attr_name}")
        generator_train = hf_dataset_to_generator(dataset,**kwargs) 
        gt_train = hf_dataset_to_generator(dataset,**{**kwargs,"dataset_column": kwargs.get("ground_truth_column", "ground_truth"),
                                                      "preprocess_fn" : kwargs.get("gt_processing_fn", None)}) 
        if kwargs.get("use_val", False):
            generator_val = hf_dataset_to_generator(dataset,split=kwargs.get("val_split","validation"),**kwargs)
            gt_val = hf_dataset_to_generator(dataset,split=kwargs.get("val_split","validation"),
                                             **{**kwargs,"dataset_column": kwargs.get("ground_truth_column", "ground_truth"),
                                               "preprocess_fn" : kwargs.get("gt_processing_fn", None)})
            buffer_val_gt = BatchIterator(gt_val,kwargs.get("out_batch_size", 1))
            
        d_sub = kwargs.pop("d_submodule", kwargs.pop("d_submodule", None))
        training_device = kwargs.get("training_device", "cpu")
        autocast_dtype = kwargs.get("autocast_dtype", th.float32)
        
        use_memmap = kwargs.get("use_memmap", False)
        
        if use_memmap and os.path.exists(cache_path(dataset,"train",kwargs.get('subset',None))):
            log(f"Using existing memmap at {cache_path(dataset,'train',kwargs.get('subset',None))} for training activations")
            buffer_train = ShardedActivationMemmapDataset(cache_path(dataset,"train",kwargs.get('subset',None)),**kwargs)
        elif use_memmap:
            buffer_train = TextImageActivationBuffer(generator_train, self.model, accessor,d_submodule=d_sub, **kwargs) 
            log(f"Creating memmap at {cache_path(dataset,'train',kwargs.get('subset',None))} for training activations")
            buffer_train = convert_buffer_to_memap(buffer_train,memmap_dir=cache_path(dataset,"train",kwargs.get('subset',None)), **kwargs)
        else:
            buffer_train = TextImageActivationBuffer(generator_train, self.model, accessor,d_submodule=d_sub, **kwargs)
        
        if use_memmap and os.path.exists(cache_path(dataset,"val",kwargs.get('subset',None))) and kwargs.get("use_val", False):
            log(f"Using existing memmap at {cache_path(dataset,'val',kwargs.get('subset',None))} for validation activations")
            buffer_val = ShardedActivationMemmapDataset(cache_path(dataset,"val",kwargs.get('subset',None)),**kwargs)
        elif use_memmap and kwargs.get("use_val", False):
            buffer_val = TextImageActivationBuffer(generator_val, self.model, accessor,d_submodule=d_sub, **kwargs)
            log(f"Creating memmap at {cache_path(dataset,'val',kwargs.get('subset',None))} for validation activations")
            buffer_val = convert_buffer_to_memap(buffer_val,memmap_dir=cache_path(dataset,"val",kwargs.get('subset',None)),**kwargs)
        elif kwargs.get("use_val", False):
            buffer_val = TextImageActivationBuffer(generator_val, self.model, accessor,d_submodule=d_sub, **kwargs)
        
        use_cache = kwargs.get("cache_activations", False)
        if use_cache:
            buffer_train = CachedActivationIterator(buffer_train, **kwargs)
            if kwargs.get("use_val", False):    
                buffer_val = CachedActivationIterator(buffer_val, **kwargs)
            
        buffer_train_gt  = BatchIterator(gt_train,kwargs.get("out_batch_size", 1))
         
        log_steps = kwargs.get("log_steps", 1) 
        steps = kwargs.get("train_steps", 1) 
        # autocast_context = th.autocast(device_type=training_device, dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
        if optimizers is None:
            optimizers = [th.optim.Adam(mapper.parameters(), lr=kwargs.get("lr",1e-5))]
        mapper = mapper.to(device=training_device, dtype=autocast_dtype) 
        
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)  
        
        val_losses = [] 
        step = 0
        log(f"Beginning training for {steps} steps")
        while True:
            for act,gt in zip(iter(buffer_train),iter(buffer_train_gt)):
                act = act.to(training_device, dtype=autocast_dtype)
                if type(gt) is list:
                    gt = th.stack(gt, dim=0)
                gt = gt.to(training_device)
                
                mapped = mapper(act) 
                loss = loss_fn(mapped, gt)
                loss.backward()
            
                for opt in optimizers:
                    opt.step()
                    opt.zero_grad()

                if log_steps and step % log_steps == 0:
                    # eval
                    val_loss=0
                    n_samples=0
                    if kwargs.get("use_val", False):
                        with th.no_grad():                   
                            for val_act,gt_val in zip(iter(buffer_val),iter(buffer_val_gt)):
                                val_act = val_act.to(device=training_device, dtype=autocast_dtype)
                                # gt_val = gt_val.to(training_device)
                                if type(gt_val) is list:
                                    gt_val = th.stack(gt_val, dim=0).to(training_device)
                         
                                mapped_val = mapper(val_act)
                                val_loss += loss_fn(mapped_val, gt_val)
                                n_samples += 1
      
                            val_loss = val_loss / n_samples
                            if val_loss < min(val_losses, default=float('inf')):
                                best_mapper = mapper.state_dict()
                            val_losses.append(val_loss)
                        update = TrainUpdate(step=step, parts={"loss": loss.item(), "val_loss": val_loss.item()})    
                    else:
                        update = TrainUpdate(step=step, parts={"loss": loss.item()})
                    yield update
                step += 1    
                if step >= steps:
                    break
            if step >= steps:
                    break
            
        if kwargs.get("use_val", False):
           assert "best_mapper" in locals(), "No best mapper found during validation"
           mapper = mapper.load_state_dict(best_mapper) 
        log("Finished training KSteer mapper")       
        self.classifier = mapper 
        if not hasattr(self, "out"):
            self.out = Output()
        self.out.run_metadata = {**kwargs}
        self.out.best_ckpt = best_mapper
        return self.out

    @th.no_grad()
    def predict_proba(self, X: th.Tensor) -> np.ndarray:
        self.classifier.eval()
        X = X.to(self.model.device)
        logits = self.classifier(X)
        probs = th.sigmoid(logits)
        return probs.cpu().numpy()
    
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
        target_idx: List[int],
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
            loss_vec = self.compute_steering_loss(
                logits, target_idx=target_idx, avoid_idx=avoid_idx
            )
            loss = loss_vec.mean()
            grads = th.autograd.grad(loss, curr, retain_graph=False)[0]
            current_alpha = alpha * (step_size_decay ** step)
            steered = (curr - current_alpha * grads).detach()
        
        th.set_grad_enabled(False)    
        return steered
    
    def eval(self, *args,**kwargs) -> None:
        pass
    
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

        
        