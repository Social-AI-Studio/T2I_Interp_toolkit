from t2Interp.accessors import ModuleAccessor
from abc import ABC, abstractmethod
from loguru import logger
from typing import Any, Dict, Optional, List, Callable, Union
from dictionary_learning.utils import hf_dataset_to_generator
from utils.utils import batchify
from utils.metrics import MetricBase
import torch as th
from utils.output import Output 
from t2Interp.intervention import run_intervention, SteeringIntervention
from utils.buffer import t2IActivationBuffer
from utils.text_image_buffer import TextImageActivationBuffer
from contextlib import nullcontext
from utils.runningstats import TrainUpdate
import numpy as np
    
class Steer(ABC):
    @abstractmethod
    def fit(self, dataset:dict, accessors:ModuleAccessor,**kwargs) -> None:
        pass
    
    @abstractmethod
    def eval(self, *args,**kwargs) -> None:
        pass
    
    @abstractmethod
    def steer(self, *args,**kwargs) -> Any:
        pass
    
class KSteer(Steer):
    def __init__(self, model):
        self.model = model
        
    def fit(self,dataset, accessor, mapper:th.nn.Module,loss_fn: Optional[Callable] = None,optimizers: List[th.optim.Optimizer]=None, **kwargs):
        generator_train = hf_dataset_to_generator(dataset,**kwargs) 
        gt_train = hf_dataset_to_generator(dataset,**{**kwargs,"dataset_column": kwargs.get("ground_truth_column", "ground_truth"),
                                                      "preprocess_fn" : kwargs.get("gt_processing_fn", None)}) 
        if kwargs.get("use_val", False):
            generator_val = hf_dataset_to_generator(dataset,split=kwargs.get("val_split","validation"),**kwargs)
            gt_val = hf_dataset_to_generator(dataset,split=kwargs.get("val_split","validation"),
                                             **{**kwargs,"dataset_column": kwargs.get("ground_truth_column", "ground_truth"),
                                               "preprocess_fn" : kwargs.get("gt_processing_fn", None)})
            buffer_val_gt = batchify(gt_val,kwargs.get("out_batch_size", 1))
            
        d_sub = kwargs.pop("d_submodule", kwargs.pop("d_submodule", None))
        training_device = kwargs.get("training_device", "cpu")
        autocast_dtype = kwargs.get("autocast_dtype", th.float32)
        buffer_train = TextImageActivationBuffer(generator_train, self.model, accessor,d_submodule=d_sub, **kwargs) 
        buffer_val = TextImageActivationBuffer(generator_val, self.model, accessor,d_submodule=d_sub, **kwargs) 
        buffer_train_gt  = batchify(gt_train,kwargs.get("out_batch_size", 1))
         
        log_steps = kwargs.get("log_steps", 1) 
        steps = kwargs.get("steps", 1) 
        # autocast_context = th.autocast(device_type=training_device, dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
        if optimizers is None:
            optimizers = [th.optim.Adam(mapper.parameters(), lr=kwargs.get("lr",1e-5))]
        mapper = mapper.to(device=training_device, dtype=autocast_dtype) 
        
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)  
        
        val_losses = [] 
        for step, (act,gt) in enumerate(zip(buffer_train,buffer_train_gt)):
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
                if kwargs.get("use_val", False):
                    with th.no_grad():                            
                        for val_act,gt_val in zip(buffer_val,buffer_val_gt):
                            val_act = val_act.to(device=training_device, dtype=autocast_dtype)
                            # gt_val = gt_val.to(training_device)
                            if type(gt_val) is list:
                                gt_val = th.stack(gt_val, dim=0).to(training_device)
                                mapped_val = mapper(val_act)
                                val_loss += loss_fn(mapped_val, gt_val)
                        val_loss = val_loss.mean(dim=0).item() 
                        if val_loss < min(val_losses, default=float('inf')):
                            best_mapper = mapper.state_dict()
                        val_losses.append(val_loss)
                    update = TrainUpdate(step=step, parts={"loss": loss.item(), "val_loss": val_loss.item()})    
                else:
                    update = TrainUpdate(step=step, parts={"loss": loss.item()})
                yield update
                
            if step >= steps:
                break
        if kwargs.get("use_val", False):
           assert "best_mapper" in locals(), "No best mapper found during validation"
           mapper = mapper.load_state_dict(best_mapper)     
        self.classifier = mapper 

    @th.no_grad()
    def predict_proba(self, X: th.Tensor) -> np.ndarray:
        self.classifier.eval()
        X = X.to(self.model.device)
        logits = self.classifier(X)
        probs = th.sigmoid(logits)
        return probs.cpu().numpy()
    
    def compute_steering_loss(
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
                target_term = logits[:, target_idx].mean(dim=1)
            else:
                target_term = th.zeros(B, device=logits.device)
            return avoid_term - target_term
    
    @th.no_grad()
    def steer(
        self,
        acts: Union[np.ndarray, th.Tensor],
        target_idx: List[int],
        avoid_idx: List[int] | None = None,
        *,
        alpha: float = 1.0,
        steps: int = 1,
        step_size_decay: float = 1.0,
    ) -> th.Tensor:
        if avoid_idx is None:
            avoid_idx = []
        if isinstance(acts, np.ndarray):
            acts_t = th.as_tensor(acts, dtype=th.float32, device=self.device)
        else:
            acts_t = acts.to(self.device, dtype=th.float32)

        steered = acts_t.detach().clone()
        for step in range(steps):
            curr = steered.clone().requires_grad_(True)
            logits = self.classifier(curr)
            loss_vec = self.compute_steering_loss(
                logits, target_idx=target_idx, avoid_idx=avoid_idx
            )
            loss = loss_vec.mean()
            grads = th.autograd.grad(loss, curr, retain_graph=False)[0]
            current_alpha = alpha * (step_size_decay ** step)
            steered = (curr - current_alpha * grads).detach()
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
        for b in batchify(gen,batch_size):          
            with self.model.generate(b, num_inference_steps=num_inference_steps, seed=seed, **generate_kwargs) as tracer:
                for accessor in accessors:
                    act = accessor.value
                    pos_activations[accessor.attr_name].append(act.cpu())
                tracer.stop()
                
        # get negative activations
        neg_activations = {accessor.attr_name:[] for accessor in accessors}
        gen = hf_dataset_to_generator(dataset,**{"dataset_column": "negative_prompt"})
        for b in batchify(gen,batch_size):          
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
        metrics={}
        for attr_name,dir in self.directions.items():
            logger.info(f"Evaluating direction for {dir}")
            output = run_intervention(model=self.model, prompts=eval_prompts, n_steps = 1, start_step=0, end_step=1, seed=40, interventions = [SteeringIntervention(dir)], **kwargs)
            metrics[attr_name]=metric.compute(output)    
        if kwargs.get("maximize_metric", True):
            self.direction= self.directions[max(metrics, key=metrics.get)]    
        else:    
            self.direction= self.directions[min(metrics, key=metrics.get)] 
    
    def steer(self,**kwargs):
        pass 
    
class ConceptSearch:
    def __init__(self, model):
        self.model = model

    def search_and_steer(self, dataset:dict, accessors:ModuleAccessor, steering_type:Steer, metric:MetricBase, eval_prompts, **kwargs):
        steering_type.find_directions(dataset, accessors, **kwargs)
        steering_type.eval(metric, eval_prompts, **kwargs)  
        steering_type.steer(**kwargs)
        
        