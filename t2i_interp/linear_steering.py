from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from copy import deepcopy
from typing import Any
import os
import numpy as np
import torch as th

from dictionary_learning.utils import hf_dataset_to_generator
from t2i_interp.accessors.accessor import ModuleAccessor
from t2i_interp.t2i import T2IModel
from t2i_interp.utils.metrics import MetricBase
from t2i_interp.utils.output import Output
from t2i_interp.utils.runningstats import TrainUpdate, Update
from t2i_interp.utils.utils import (
    ActivationConfig,
    BatchIterator,
    normalize_batch,
)
from t2i_interp.utils.T2I.hook import UNetAlterHook, TextEncoderAlterHook
from t2i_interp.utils.trace import TraceDict
from functools import partial


class Steer(ABC):
    @abstractmethod
    def fit(self, model: T2IModel, dataset: dict, accessors: ModuleAccessor, **kwargs) -> None:
        pass

    @abstractmethod
    def eval(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def steer(self, *args, **kwargs) -> Any:
        pass


class KSteer(Steer):
    def __init__(self, model: T2IModel = None):
        """Initialize KSteer steering mechanism.

        Args:
            model: T2IModel instance (can also be provided in fit())
        """
        self.model = model

    def fit(
        self,
        train_loader: Any,
        mapper: th.nn.Module,
        val_loader: Any | None = None,
        loss_fn: Callable | None = None,
        optimizers: list[th.optim.Optimizer] | None = None,
        out: Output | None = None,
        model: T2IModel | None = None,
        **kwargs,
    ) -> Generator[Update, None, Output]:
        """Train a KSteer classifier mapper on model activations.

        Args:
            train_loader: ActivationsDataloader for training
            mapper: Neural network mapper (e.g., MLPMapper) to train
            val_loader: Optional ActivationsDataloader for validation
            loss_fn: Loss function (e.g., CrossEntropyLoss)
            optimizers: List of optimizers
            out: Output object
            model: T2IModel instance
            **kwargs: Config overrides


        Yields:
            Update: Training progress updates containing:
                - info (str): Informational messages
                - TrainUpdate: Training metrics (step, loss, val_loss)

        Returns:
            Output: Training results containing:
                - run_metadata (dict): Training configuration
                - best_ckpt (dict): Best model checkpoint state_dict

        Raises:
            AssertionError: If model is not provided at init or in fit()
        """

        # resolve model/out
        if self.model is None and model is not None:
            self.model = model
        assert self.model is not None, "Model must be provided at init or in fit()"
        if out is not None:
            self.out = out
        else:
            self.out = Output()


        
        # Load config
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "config/mapper_training_config.yaml")
        with open(config_path, "r") as f:
            default_cfg = yaml.safe_load(f)
        
        cfg_dict = default_cfg.copy()
        for k, v in kwargs.items():
            if k in cfg_dict:
                if isinstance(cfg_dict[k], dict) and isinstance(v, dict):
                    cfg_dict[k].update(v)
                else:
                    cfg_dict[k] = v
            else:
                 cfg_dict[k] = v

        cfg = ActivationConfig(
            steps=cfg_dict.get("train_steps"),
            log_steps=cfg_dict.get("log_steps"),
            lr=cfg_dict.get("lr"),
            training_device=cfg_dict.get("training_device"),
            autocast_dtype=cfg_dict.get("autocast_dtype"),
            grad_clip_norm=cfg_dict.get("grad_clip_norm"),
            data_loader_kwargs=cfg_dict.get("data_loader_kwargs", {}),
        )

        columns = cfg_dict.get("columns", ["label"])
        if "target_column" in kwargs:
             columns = [kwargs["target_column"]]
        
        # optim
        if optimizers is None:
            optimizers = [th.optim.Adam(mapper.parameters(), lr=cfg.lr)]
        mapper = mapper.to(device=cfg.training_device, dtype=getattr(th, cfg.autocast_dtype) if isinstance(cfg.autocast_dtype, str) else cfg.autocast_dtype)

        # freeze model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # logging
        yield Update(
            info=f"Starting KSteer training"
        )
        yield Update(
            info=f"""Train steps={cfg.steps}, device={cfg.training_device},
                     dtype={cfg.autocast_dtype},memmap={cfg.data_loader_kwargs.get("use_memmap", False)},
                     cached={cfg.data_loader_kwargs.get("cache_activations", False)}"""
        )

        # best ckpt
        best_mapper_sd = deepcopy(mapper.state_dict())
        best_val = float("inf")

        # training loop
        step = 0
        while step < cfg.steps:
            # ActivationsDataloader is infinite loop by default (repeat=True)
            train_iter = train_loader.iterate()
            
            for batch_data in train_iter:
                if isinstance(batch_data, (tuple, list)):
                    # (act, extra1, extra2...)
                    act = batch_data[0]
                    # We pass extra_keys=[..., ...]. They come back in order.
                    # columns=["label"] -> extra_keys=["label.pth"] -> batch_data[1] is label
                    if len(batch_data) > 1:
                        if len(batch_data) == 2:
                             gt = batch_data[1]
                        else:
                             gt = list(batch_data[1:])
                    else:
                        gt = None
                else:
                    act = batch_data
                    gt = None

                act = act.to(cfg.training_device, dtype=getattr(th, cfg.autocast_dtype) if isinstance(cfg.autocast_dtype, str) else cfg.autocast_dtype)
                act = act.to(cfg.training_device, dtype=getattr(th, cfg.autocast_dtype) if isinstance(cfg.autocast_dtype, str) else cfg.autocast_dtype)
                
                # Normalize targets
                gt = normalize_batch(gt, cfg.training_device) if gt is not None else None

                # Normalize outputs (mapper returns tensor or tuple)
                mapped = normalize_batch(mapper(act), cfg.training_device)

                if gt is not None:
                    # Both are lists of tensors now
                    loss = sum(loss_fn(m, g) for m, g in zip(mapped, gt))
                else:
                    # Unsupervised or custom case
                    # Unwrap if single item to behave like standard loss input
                    inputs = mapped[0] if len(mapped) == 1 else mapped
                    loss = loss_fn(inputs, None)

                loss.backward()
                if cfg.grad_clip_norm is not None:
                    for opt in optimizers:
                        th.nn.utils.clip_grad_norm_(mapper.parameters(), cfg.grad_clip_norm)

                for opt in optimizers:
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                # logging & validation
                if cfg.log_steps and (step % cfg.log_steps == 0):
                    if cfg.data_loader_kwargs.get("use_val", False) and val_loader is not None:
                        mapper.eval()
                        with th.no_grad():
                            val_loss, n = 0.0, 0
                            
                            val_loader.repeat = False
                            if hasattr(val_loader, "reset"):
                                val_loader.reset()
                            for batch_data in val_loader.iterate():
                                if isinstance(batch_data, (tuple, list)):
                                    val_act = batch_data[0]
                                    if len(batch_data) > 1:
                                        if len(batch_data) == 2:
                                             gt_val = batch_data[1]
                                        else:
                                             gt_val = list(batch_data[1:])
                                    else:
                                        gt_val = None

                                val_act = val_act.to(cfg.training_device, dtype=getattr(th, cfg.autocast_dtype) if isinstance(cfg.autocast_dtype, str) else cfg.autocast_dtype)
                                val_act = val_act.to(cfg.training_device, dtype=getattr(th, cfg.autocast_dtype) if isinstance(cfg.autocast_dtype, str) else cfg.autocast_dtype)
                                
                                gt_val = normalize_batch(gt_val, cfg.training_device) if gt_val is not None else None
                                mapped_val = normalize_batch(mapper(val_act), cfg.training_device)

                                if gt_val is not None:
                                    for m_val, g_val in zip(mapped_val, gt_val, strict=False):
                                        val_loss += float(loss_fn(m_val, g_val))
                                else:
                                    inputs = mapped_val[0] if len(mapped_val) == 1 else mapped_val
                                    val_loss += float(loss_fn(inputs, None))
                                n += 1
                            val_loss = val_loss / max(n, 1)

                            # best ckpt
                            if val_loss < best_val:
                                best_val = val_loss
                                best_mapper_sd = deepcopy(mapper.state_dict())
                        mapper.train()
                        yield TrainUpdate(
                            step=step,
                            parts={"loss": float(loss.item()), "val_loss": float(val_loss)},
                        )
                    else:
                        yield TrainUpdate(step=step, parts={"loss": float(loss.item())})

                step += 1
                if step >= cfg.steps:
                    break

        # restore best (if val used)
        if cfg.data_loader_kwargs.get("use_val", False):
            mapper.load_state_dict(best_mapper_sd)

        yield Update(info="Finished training KSteer mapper")
        yield Update(info="Evaluating KSteer mapper on validation set")
        val_acc = self.eval(
            val_loader,
            mapper,
            model=self.model,
            **kwargs,
        )
        yield Update(info=f"KSteer mapper validation accuracy: {val_acc:.4f}")
        # self.classifier = mapper
        # self.out.run_metadata = {**kwargs}
        # self.out.best_ckpt = best_mapper_sd
        # load best mapper
        mapper.load_state_dict(best_mapper_sd)
        self.classifier = mapper
        yield self.classifier

    @th.no_grad()
    def eval(
        self,
        val_loader: Any,
        mapper: th.nn.Module,
        out: Output | None = None,
        model: T2IModel | None = None,
        **kwargs,
    ):
        if self.model is None and model is not None:
            self.model = model
        assert self.model is not None, "Model must be provided at init or in fit()"

        # Load config
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "config/mapper_training_config.yaml")
        with open(config_path, "r") as f:
            default_cfg = yaml.safe_load(f)
        
        cfg_dict = default_cfg.copy()
        for k, v in kwargs.items():
            if k in cfg_dict:
                if isinstance(cfg_dict[k], dict) and isinstance(v, dict):
                    cfg_dict[k].update(v)
                else:
                    cfg_dict[k] = v
            else:
                 cfg_dict[k] = v

        cfg = ActivationConfig(
            data_loader_kwargs=cfg_dict.get("data_loader_kwargs", {}),
             training_device=cfg_dict.get("training_device", "cpu"),
             autocast_dtype=cfg_dict.get("autocast_dtype", th.float32),
        )
        
        val_loader.repeat = False
        if hasattr(val_loader, "reset"):
            val_loader.reset()

        mapper.eval()
        # freeze model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # logging
        # yield Update(info=f"Evaluating KSteer mapper on dataset={dataset} accessor={accessor.attr_name}")

        val_correct = 0
        val_total = 0
        
        for batch_data in val_loader.iterate():
            if isinstance(batch_data, (tuple, list)):
                val_act = batch_data[0]
                if len(batch_data) > 1:
                    if len(batch_data) == 2:
                         gt_val = batch_data[1]
                    else:
                         gt_val = list(batch_data[1:])
                else:
                    gt_val = None
            else:
                 val_act = batch_data
                 gt_val = None

            val_act = val_act.to(cfg.training_device, dtype=getattr(th, cfg.autocast_dtype) if isinstance(cfg.autocast_dtype, str) else cfg.autocast_dtype)
            
            gt_val = normalize_batch(gt_val, cfg.training_device) if gt_val is not None else None
            mapped_val = normalize_batch(mapper(val_act), cfg.training_device)

            if gt_val is not None:
                for m_val, g_val in zip(mapped_val, gt_val):
                    _, predicted = th.max(m_val, 1)
                    val_correct += (predicted == g_val).sum().item()
                    val_total += g_val.shape[0]
            else:
                # Without GT, cannot compute accuracy
                pass

        # if val_total == 0:
        #     return 0.0
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

        target_indices = [i for i, logits in enumerate(logits) if logits[:, i].mean() < mean_logits]
        avoid_indices = [i for i, logits in enumerate(logits) if logits[:, i].mean() > mean_logits]

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
        target_idx: list[int] | th.Tensor,
        avoid_idx: list[int] | th.Tensor,
    ) -> th.Tensor:
        if not th.is_tensor(target_idx):
            target_idx = th.as_tensor(target_idx, device=logits.device)
        else:
            target_idx = target_idx.to(logits.device)
        if not th.is_tensor(avoid_idx):
            avoid_idx = th.as_tensor(avoid_idx, device=logits.device)
        else:
            avoid_idx = avoid_idx.to(logits.device)

        if logits.ndim > 2:
            logits = logits.reshape(-1, logits.shape[-1])
            
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

    def apply_steering(
        self,
        acts: np.ndarray | th.Tensor,
        target_idx: list[int] | None = None,
        avoid_idx: list[int] | None = None,
        *,
        alpha: float = 1,
        steer_steps: int = 1,
        step_size_decay: float = 1.0,
        mapper: str | None = None,
        **kwargs,
    ) -> th.Tensor:
        """Apply gradient-based steering to activations."""
        th.set_grad_enabled(True)

        if not hasattr(self, "classifier") and mapper is not None:
            self.classifier = mapper

        assert hasattr(
            self, "classifier"
        ), "Classifier not found. Please fit the model or provide a classifier_path."

        if avoid_idx is None:
            avoid_idx = []
        
        # Get classifier dtype
        param = next(self.classifier.parameters())
        dtype = param.dtype
        device = param.device

        if isinstance(acts, np.ndarray):
            acts_t = th.as_tensor(acts, dtype=dtype, device=device)
        else:
            acts_t = acts.to(device, dtype=dtype)

        # Clone and require grad
        steered = acts_t.detach().clone()
        
        for step in range(steer_steps):
            curr = steered.clone().requires_grad_(True)
            logits = self.classifier(curr)
            
            if isinstance(logits, (tuple, list)):
                loss = 0
                for i, logit_head in enumerate(logits):
                    # Determine target/avoid for this head
                    tgt = None
                    if target_idx is not None:
                         if isinstance(target_idx, list) and len(target_idx) > i and isinstance(target_idx[i], (list, int, th.Tensor)):
                             tgt = target_idx[i]
                         elif i == 0 and isinstance(target_idx, (int, th.Tensor)): 
                             # fallback for single target passed to multihead
                             pass
                    
                    avd = None
                    if avoid_idx is not None:
                         if isinstance(avoid_idx, list) and len(avoid_idx) > i:
                             avd = avoid_idx[i]

                    if tgt is not None or avd is not None:
                         if tgt is None: tgt = []
                         if avd is None: avd = []
                         l = self.compute_steering_loss(logit_head, target_idx=tgt, avoid_idx=avd)
                         loss += l.mean()
            else:
                loss_vec = self.compute_steering_loss(logits, target_idx=target_idx, avoid_idx=avoid_idx)
                loss = loss_vec.mean()

            grads = th.autograd.grad(loss, curr, retain_graph=False)[0]
            current_alpha = alpha * (step_size_decay**step)
            steered = (curr - current_alpha * grads).detach()

        th.set_grad_enabled(False)
        
        # Cast back to input dtype if it was a tensor
        if th.is_tensor(acts):
            steered = steered.to(dtype=acts.dtype)
            
        return steered

    def steer(self, prompts, layer_name, target_idx, avoid_idx, alpha, steer_steps, **kwargs) -> Any:
        hook = UNetAlterHook(
            policy=partial(self.apply_steering, target_idx=target_idx, avoid_idx=avoid_idx, alpha=alpha, steer_steps=steer_steps)
        )
        module = self.model.resolve_accessor(layer_name).module
        with TraceDict([module], hook):
            imgs = self.model.pipeline(prompts, num_inference_steps=kwargs.get("num_inference_steps", 50)).images
        return imgs



class CAA(Steer):
    def __init__(self, model: T2IModel = None):
        self.model = model
        self.steering_vecs = {}  # {layer_name: tensor}

    def fit(
        self,
        pos_acts: th.Tensor | list[th.Tensor],
        neg_acts: th.Tensor | list[th.Tensor] | None = None,
        attr_name: str = "default",
        **kwargs,
    ):
        """
        Compute steering vector = Mean(Pos) - Mean(Neg).
        
        Args:
            pos_acts: Positive activations (tensor or list of tensors).
            neg_acts: Negative activations (tensor or list of tensors).
            layer_name: Name of the layer/direction.
        """
        if isinstance(pos_acts, list):
            pos_acts = th.stack(pos_acts) if len(pos_acts) > 0 else th.tensor([])
        if neg_acts is not None and isinstance(neg_acts, list):
            neg_acts = th.stack(neg_acts) if len(neg_acts) > 0 else th.tensor([])
            
        if pos_acts.numel() == 0:
             raise ValueError("No positive activations provided")

        mean_pos = pos_acts.mean(dim=0, keepdim=True)
        
        if neg_acts is not None and neg_acts.numel() > 0:
            mean_neg = neg_acts.mean(dim=0, keepdim=True)
        else:
            mean_neg = th.zeros_like(mean_pos)
        print(mean_pos.shape, mean_neg.shape)    
        steering_vec = (mean_pos - mean_neg).detach()
        steering_vec = steering_vec / (steering_vec.norm(p=2) + 1e-12)

        self.steering_vecs[attr_name] = steering_vec
        return steering_vec

    def apply_steering(
        self, 
        acts: th.Tensor, 
        steering_vecs: th.Tensor | None = None, 
        alphas: list[float] | None = None, 
        **kwargs
    ):
        """
        Add steering vector to activations.
        acts: (B, ...)
        """

        if steering_vecs is None:
            steering_vecs = self.steering_vecs
        
        assert steering_vecs is not None

        if alphas is None:
            alphas = [1.0] * len(steering_vecs)
        # zip and apply steering vecs, alphas
        acts += sum([alpha * vec for vec, alpha in zip(steering_vecs, alphas)]) 
        return acts
    
    def steer(self, prompts, layer_name, steering_vecs, alphas, **kwargs):
        hook = UNetAlterHook(
            policy=partial(self.apply_steering, steering_vecs=steering_vecs, alphas=alphas)
        )
        module = self.model.resolve_accessor(layer_name).module
        with TraceDict([module], hook):
            imgs = self.model.pipeline(prompts, num_inference_steps=kwargs.get("num_inference_steps", 50)).images
        return imgs
    
    def eval(self, *args, **kwargs):
        pass

class LoREEFT(Steer):
    def __init__(self, model: T2IModel = None):
        """Initialise LoREEFT steering mechanism.

        Args:
            model: :class:`T2IModel` instance (can also be provided in
                :meth:`fit`).
        """
        self.model = model
        self.loreft = None  # trained LoReFTLayer stored after fit()

    # ------------------------------------------------------------------
    # Internal training routines
    # ------------------------------------------------------------------

    def train_loreft_on_clip(
        self,
        loader,                    # DataLoader: batches = {"teacher": token_dict, "base": token_dict}
        val_loader=None,           # Optional DataLoader with same format for validation
        layer_idx: int = 5,
        rank: int = 16,
        num_steps: int = 10_000,
        lr: float = 1e-3,
        device: str = "cuda:0",
        log_steps: int = 100,
    ) -> Generator[Update, None, "LoReFTLayer"]:
        """Train LoReFT on a CLIP text-encoder hidden layer.

        The CLIP text encoder is extracted from ``self.model.pipeline`` and
        frozen; only the LoReFT parameters are optimised.

        Args:
            loader: Pre-built DataLoader.  Each batch must have the shape
                ``{"teacher": token_dict, "base": token_dict}`` where each
                token dict contains at least ``input_ids`` and
                ``attention_mask``.
            layer_idx: Index into the encoder's ``hidden_states`` tuple
                (0 = embedding layer, 1 = first transformer layer, …).
            rank: Rank of the LoReFT decomposition.
            num_steps: Total optimisation steps.
            lr: AdamW learning rate.
            device: Compute device (e.g. ``"cuda:0"``).
            log_steps: Yield a :class:`TrainUpdate` every this many steps.

        Yields:
            :class:`Update` / :class:`TrainUpdate` progress events.

        Returns:
            Trained :class:`LoReFTLayer`.
        """
        from t2i_interp.loreft import LoReFTLayer

        pipe         = self.model.pipeline
        text_encoder = pipe.text_encoder.to(device)
        text_encoder.eval()
        for p in text_encoder.parameters():
            p.requires_grad_(False)

        d_model = text_encoder.config.hidden_size
        yield Update(
            info=f"Starting LoREEFT-CLIP training: d_model={d_model}, "
                 f"layer_idx={layer_idx}, rank={rank}, steps={num_steps}, "
                 f"val={'yes' if val_loader is not None else 'no'}"
        )

        loreft         = LoReFTLayer(d_model=d_model, rank=rank).to(device)
        optimizer      = th.optim.AdamW(loreft.parameters(), lr=lr)
        best_val       = float("inf")
        best_loreft_sd = deepcopy(loreft.state_dict())

        def _clip_loss(h_base_b, h_teacher_b, mask_b):
            """MSE on active tokens, returning a scalar."""
            diff = (loreft(h_base_b) - h_teacher_b) ** 2
            return (diff * mask_b).sum() / mask_b.sum().clamp(min=1.0)

        def _run_val():
            loreft.eval()
            val_loss, n = 0.0, 0
            with th.no_grad():
                for vbatch in val_loader:
                    vt = {k: v.to(device) for k, v in vbatch["teacher"].items()}
                    vb = {k: v.to(device) for k, v in vbatch["base"].items()}
                    ot = text_encoder(**vt, output_hidden_states=True, return_dict=True)
                    ob = text_encoder(**vb, output_hidden_states=True, return_dict=True)
                    ht = ot.hidden_states[layer_idx].to(th.float32)
                    hb = ob.hidden_states[layer_idx].to(th.float32)
                    vm = vb["attention_mask"].to(th.float32).unsqueeze(-1)
                    val_loss += float(_clip_loss(hb, ht, vm))
                    n += 1
            loreft.train()
            return val_loss / max(n, 1)

        step = 0
        it   = iter(loader)
        while step < num_steps:
            try:
                batch = next(it)
            except StopIteration:
                it    = iter(loader)
                batch = next(it)

            teacher_tokens = {k: v.to(device) for k, v in batch["teacher"].items()}
            base_tokens    = {k: v.to(device) for k, v in batch["base"].items()}

            with th.no_grad():
                out_teacher = text_encoder(
                    **teacher_tokens, output_hidden_states=True, return_dict=True
                )
                out_base = text_encoder(
                    **base_tokens, output_hidden_states=True, return_dict=True
                )
                h_teacher = out_teacher.hidden_states[layer_idx].to(th.float32)  # (B, T, D)
                h_base    = out_base.hidden_states[layer_idx].to(th.float32)

            attention_mask = base_tokens["attention_mask"].to(th.float32).unsqueeze(-1)  # (B, T, 1)

            h_edit = loreft(h_base)
            diff   = (h_edit - h_teacher) ** 2
            loss   = (diff * attention_mask).sum() / attention_mask.sum().clamp(min=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if log_steps and (step % log_steps == 0):
                if val_loader is not None:
                    val_loss = _run_val()
                    if val_loss < best_val:
                        best_val       = val_loss
                        best_loreft_sd = deepcopy(loreft.state_dict())
                    yield TrainUpdate(step=step, parts={"loss": float(loss.item()), "val_loss": val_loss})
                else:
                    yield TrainUpdate(step=step, parts={"loss": float(loss.item())})
            step += 1

        # restore best checkpoint
        if val_loader is not None:
            loreft.load_state_dict(best_loreft_sd)
            yield Update(info=f"LoREEFT-CLIP training done. Best val_loss={best_val:.6f}")
        else:
            yield Update(info="LoREEFT-CLIP training done.")
        return loreft

    def train_loreft_on_unet(
        self,
        loader,                    # DataLoader: batches = {"teacher": prompts, "base": prompts}
        val_loader=None,           # Optional DataLoader with same format for validation
        layer_name: str = "",     # dot-path to the target UNet module
        rank: int = 16,
        d_model: int = 1_024,
        num_steps: int = 5_000,
        lr: float = 1e-3,
        device: str = "cuda:0",
        log_steps: int = 100,
        **gen_config,
    ) -> Generator[Update, None, "LoReFTLayer"]:
        """Train LoReFT on a UNet hidden layer via forward-pass hooks.

        Teacher and base activations are captured by running the full UNet
        diffusion pipeline with per-module hooks.

        Args:
            loader: Pre-built DataLoader.  Each batch must have the shape
                ``{"teacher": prompts, "base": prompts}`` where ``prompts``
                are string lists or pre-tokenised dicts accepted by the
                pipeline.
            val_loader: Optional validation DataLoader (same format as
                ``loader``).  When provided, val loss is computed every
                ``log_steps`` steps and the best checkpoint is saved.
            layer_name: Dot-path (or accessor path) identifying the UNet
                module to hook, forwarded to
                :meth:`T2IModel.resolve_accessor`.
            rank: Rank of the LoReFT decomposition.
            d_model: Hidden dimension of the target UNet layer.
            num_steps: Total optimisation steps.
            lr: AdamW learning rate.
            device: Compute device.
            log_steps: Yield a :class:`TrainUpdate` every this many steps.
            **gen_config: Extra kwargs forwarded to ``run_with_hook``
                (e.g. ``num_inference_steps``).

        Yields:
            :class:`Update` / :class:`TrainUpdate` progress events.

        Returns:
            Trained :class:`LoReFTLayer`.
        """
        from t2i_interp.loreft import LoReFTLayer

        pipe   = self.model.pipeline
        module = self.model.resolve_accessor(layer_name).module
        yield Update(
            info=f"Starting LoREEFT-UNet training: layer={layer_name}, d_model={d_model}, "
                 f"rank={rank}, steps={num_steps}, val={'yes' if val_loader is not None else 'no'}"
        )

        loreft         = LoReFTLayer(d_model=d_model, rank=rank).to(device)
        optimizer      = th.optim.AdamW(loreft.parameters(), lr=lr)
        best_val       = float("inf")
        best_loreft_sd = deepcopy(loreft.state_dict())

        def _capture(prompts):
            """Run the pipeline and return the hooked activations."""
            hook_obj = CaptureOutputHook(denoiser_steps=None, reduce_fn=None)
            _, acts  = run_with_hook(
                pipe=pipe,
                batch={"prompt": prompts},
                module=module,
                hook_obj=hook_obj,
                **gen_config,
            )
            return acts.to(th.float32)

        def _run_val():
            loreft.eval()
            val_loss, n = 0.0, 0
            with th.no_grad():
                for vbatch in val_loader:
                    ht = _capture(vbatch["teacher"])
                    hb = _capture(vbatch["base"])
                    diff = (loreft(hb) - ht) ** 2
                    val_loss += float(diff.sum() / diff.numel())
                    n += 1
            loreft.train()
            return val_loss / max(n, 1)

        step = 0
        it   = iter(loader)
        while step < num_steps:
            try:
                batch = next(it)
            except StopIteration:
                it    = iter(loader)
                batch = next(it)

            with th.no_grad():
                h_teacher = _capture(batch["teacher"])
                h_base    = _capture(batch["base"])

            h_edit = loreft(h_base)
            diff   = (h_edit - h_teacher) ** 2
            loss   = diff.sum() / diff.numel()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if log_steps and (step % log_steps == 0):
                if val_loader is not None:
                    val_loss = _run_val()
                    if val_loss < best_val:
                        best_val       = val_loss
                        best_loreft_sd = deepcopy(loreft.state_dict())
                    yield TrainUpdate(step=step, parts={"loss": float(loss.item()), "val_loss": val_loss})
                else:
                    yield TrainUpdate(step=step, parts={"loss": float(loss.item())})
            step += 1

        # restore best checkpoint
        if val_loader is not None:
            loreft.load_state_dict(best_loreft_sd)
            yield Update(info=f"LoREEFT-UNet training done. Best val_loss={best_val:.6f}")
        else:
            yield Update(info="LoREEFT-UNet training done.")
        return loreft

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        loader,                          # pre-built train DataLoader
        layer_name: str,                 # "clip" (or contains "clip") → CLIP branch; else UNet
        val_loader=None,                 # optional validation DataLoader
        rank: int = 16,
        num_steps: int = 5_000,
        lr: float = 1e-3,
        device: str = "cuda:0",
        log_steps: int = 100,
        model: T2IModel | None = None,
        # CLIP-specific
        layer_idx: int = 5,
        # UNet-specific
        d_model: int = 1_024,
        **kwargs,
    ) -> Generator[Update, None, "LoReFTLayer"]:
        """Fit a LoReFT layer on either the CLIP encoder or a UNet module.

        Dispatches to :meth:`train_loreft_on_clip` when ``layer_name``
        contains the substring ``"clip"`` (case-insensitive), and to
        :meth:`train_loreft_on_unet` otherwise.

        Args:
            loader: Pre-built train DataLoader.  Batch format depends on the
                target branch – see the respective training methods.
            layer_name: Target layer identifier.  Use ``"clip"`` (or any
                string containing ``"clip"``) for the text encoder; for UNet
                pass the accessor dot-path to the desired module.
            val_loader: Optional validation DataLoader (same format as
                ``loader``).  When provided, val loss is computed every
                ``log_steps`` steps and the best checkpoint is restored.
            rank: LoReFT rank.
            num_steps: Optimisation steps.
            lr: Learning rate.
            device: Compute device.
            log_steps: Log / validate every this many steps.
            model: Override :class:`T2IModel` (if not provided at init).
            layer_idx: *(CLIP only)* Index into the ``hidden_states`` tuple.
            d_model: *(UNet only)* Hidden dimension of the target layer.
            **kwargs: Additional kwargs forwarded to the UNet training
                routine (e.g. ``num_inference_steps``).

        Yields:
            :class:`Update` / :class:`TrainUpdate` progress events, then
            the trained :class:`LoReFTLayer` as the final yielded value.

        Returns:
            Trained :class:`LoReFTLayer` (also stored in ``self.loreft``).
        """
        if self.model is None and model is not None:
            self.model = model
        assert self.model is not None, "Model must be provided at init or in fit()"

        if "clip" in layer_name.lower():
            train_gen = self.train_loreft_on_clip(
                loader=loader,
                val_loader=val_loader,
                layer_idx=layer_idx,
                rank=rank,
                num_steps=num_steps,
                lr=lr,
                device=device,
                log_steps=log_steps,
            )
        else:
            train_gen = self.train_loreft_on_unet(
                loader=loader,
                val_loader=val_loader,
                layer_name=layer_name,
                rank=rank,
                d_model=d_model,
                num_steps=num_steps,
                lr=lr,
                device=device,
                log_steps=log_steps,
                **kwargs,
            )

        loreft = yield from train_gen

        self.loreft = loreft
        yield loreft

    def apply_steering(
        self,
        acts: th.Tensor,
        loreft: th.nn.Module | None = None,
        **kwargs,
    ) -> th.Tensor:
        """Apply the trained LoReFT transformation to hidden-state activations.

        Args:
            acts: Hidden-state tensor ``(B, T, D)`` intercepted by the hook.
            loreft: Override the LoReFT module to apply; defaults to
                ``self.loreft``.

        Returns:
            Edited activations ``(B, T, D)``.
        """
        _loreft = loreft if loreft is not None else self.loreft
        assert _loreft is not None, (
            "No trained LoReFT module found. Call fit() first or pass loreft=."
        )
        with th.no_grad():
            return _loreft(acts.to(th.float32)).to(acts.dtype)

    def steer(
        self,
        prompts: list[str],
        layer_name: str,
        loreft: th.nn.Module | None = None,
        num_inference_steps: int = 50,
        **kwargs,
    ) -> list:
        """Generate images with LoReFT applied at the specified layer.

        Dispatches on ``layer_name``:

        * **CLIP** (``'clip'`` anywhere in ``layer_name``, case-insensitive) –
          hooks a transformer layer inside the text encoder.  ``layer_name``
          should be parseable as an integer (the layer index) or a string
          like ``"clip.5"`` / ``"clip_layer_5"``; the last integer found in
          the string is used.
        * **UNet** – hooks the UNet module resolved via
          :meth:`T2IModel.resolve_accessor`.

        Args:
            prompts: Text prompts to generate from.
            layer_name: Layer identifier – ``"clip"`` / ``"clip.5"`` for CLIP
                encoder layer 5, or a UNet accessor path.
            loreft: Override for the LoReFT module; defaults to
                ``self.loreft``.
            num_inference_steps: Diffusion steps.
            **kwargs: Forwarded to the pipeline.

        Returns:
            List of PIL images.
        """
        _loreft = loreft if loreft is not None else self.loreft
        assert _loreft is not None, (
            "No trained LoReFT module found. Call fit() first or pass loreft=."
        )

        policy = partial(self.apply_steering, loreft=_loreft)

        if "clip" in layer_name.lower():
            # Resolve the layer index from the name, e.g. "clip.5" -> 5
            import re as _re
            nums = _re.findall(r"\d+", layer_name)
            layer_idx = int(nums[-1]) if nums else 5

            pipe         = self.model.pipeline
            text_encoder = pipe.text_encoder
            # CLIP text encoder layers: text_encoder.text_model.encoder.layers
            encoder_layers = text_encoder.text_model.encoder.layers
            target_module  = encoder_layers[layer_idx]

            hook = TextEncoderAlterHook(policy=policy)
            with TraceDict([target_module], hook):
                imgs = pipe(
                    prompts,
                    num_inference_steps=num_inference_steps,
                    **kwargs,
                ).images
        else:
            module = self.model.resolve_accessor(layer_name).module
            hook   = UNetAlterHook(policy=policy)
            with TraceDict([module], hook):
                imgs = self.model.pipeline(
                    prompts,
                    num_inference_steps=num_inference_steps,
                    **kwargs,
                ).images

        return imgs

    def eval(self, *args, **kwargs):
        pass
