from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from copy import deepcopy
from typing import Any

import numpy as np
import torch as th

from dictionary_learning.utils import hf_dataset_to_generator
from t2Interp.accessors import ModuleAccessor
from t2Interp.T2I import T2IModel
from utils.metrics import MetricBase
from utils.output import Output
from utils.runningstats import TrainUpdate, Update
from utils.text_image_buffer import _build_buffer
from utils.utils import (
    ActivationConfig,
    BatchIterator,
    normalize_gt_batch,
)


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
        dataset,
        accessor,
        mapper: th.nn.Module,
        loss_fn: Callable | None = None,
        optimizers: list[th.optim.Optimizer] | None = None,
        out: Output | None = None,
        model: T2IModel | None = None,
        **kwargs,
    ) -> Generator[Update, None, Output]:
        """Train a KSteer classifier mapper on model activations.

        This method trains a neural network mapper to predict target attributes
        from model activations, enabling steering during generation.

        Args:
            dataset: HuggingFace dataset path (str) or dataset dict
            accessor: ModuleAccessor targeting the model component to extract activations from
            mapper: Neural network mapper (e.g., MLPMapper) to train
            loss_fn: Loss function (e.g., CrossEntropyLoss). If None, uses Adam default
            optimizers: List of optimizers for training. If None, creates Adam optimizer
            out: Output object to store results. If None, creates new Output()
            model: T2IModel instance. If None, uses self.model from __init__
            **kwargs: Additional configuration options:
                - train_steps (int): Number of training iterations (default: 1)
                - lr (float): Learning rate (default: 1e-5)
                - log_steps (int): Log interval (default: 1)
                - training_device (str): Device for training (default: "cpu")
                - autocast_dtype (torch.dtype): Dtype for mixed precision (default: float32)
                - grad_clip_norm (float): Gradient clipping value (default: None)
                - data_loader_kwargs (dict): Configuration for data loading:
                    - out_batch_size (int): Batch size for training
                    - use_val (bool): Whether to use validation set
                    - val_split (str): Validation split name
                    - use_memmap (bool): Use memory-mapped datasets
                    - cache_activations (bool): Cache activations in memory
                    - ground_truth_column (str): Column name for labels
                    - gt_processing_fn (Callable): Function to process labels
                    - preprocess_fn (Callable): Function to preprocess inputs

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

        Example:
            >>> from t2Interp.T2I import T2IModel
            >>> from t2Interp.concept_search import KSteer
            >>> from t2Interp.mapper import MLPMapper
            >>>
            >>> model = T2IModel("CompVis/stable-diffusion-v1-4")
            >>> accessor = model.unet.down_blocks[0].attentions[0]
            >>> mapper = MLPMapper(input_dim=4096*320, hidden_dim=4096, output_dim=7)
            >>>
            >>> ksteer = KSteer(model)
            >>> for update in ksteer.fit(
            ...     dataset="dataset/path",
            ...     accessor=accessor,
            ...     mapper=mapper,
            ...     train_steps=100,
            ...     lr=1e-5
            ... ):
            ...     print(update)
        """

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
            **{
                **cfg.data_loader_kwargs,
                "preprocess_fn": cfg.data_loader_kwargs.get("gt_processing_fn", None),
                "dataset_column": cfg.data_loader_kwargs.get("ground_truth_column", None),
            },
        )
        gt_train_iter = BatchIterator(gt_train, cfg.data_loader_kwargs.get("out_batch_size", 1))

        if cfg.data_loader_kwargs.get("use_val", False):
            gen_val = hf_dataset_to_generator(dataset, **cfg.data_loader_kwargs)
            gt_val_raw = hf_dataset_to_generator(
                dataset,
                **{
                    **cfg.data_loader_kwargs,
                    "preprocess_fn": cfg.data_loader_kwargs.get("gt_processing_fn", None),
                    "dataset_column": cfg.data_loader_kwargs.get("ground_truth_column", None),
                },
            )
            gt_val_iter = BatchIterator(gt_val_raw, cfg.data_loader_kwargs.get("out_batch_size", 1))

        # buffers
        buf_train = _build_buffer(gen_train, self.model, accessor, dataset, "train", cfg)
        buf_val = None
        if cfg.data_loader_kwargs.get("use_val", False):
            buf_val = _build_buffer(
                gen_val,
                self.model,
                accessor,
                dataset,
                cfg.data_loader_kwargs.get("split", "val"),
                cfg,
            )

        # optim
        if optimizers is None:
            optimizers = [th.optim.Adam(mapper.parameters(), lr=cfg.lr)]
        mapper = mapper.to(device=cfg.training_device, dtype=cfg.autocast_dtype)

        # freeze model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # logging
        yield Update(
            info=f"Starting KSteer training on dataset={dataset} accessor={accessor.attr_name}"
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
            # NOTE: if your iterators are finite, re-initialize per epoch
            train_iter = zip(iter(buf_train), iter(gt_train_iter), strict=False)
            for act, gt in train_iter:
                act = act.to(cfg.training_device, dtype=cfg.autocast_dtype)
                gt = normalize_gt_batch(gt, cfg.training_device)

                mapped = mapper(act)
                if isinstance(mapped, tuple):
                    mapped = list(mapped)

                if isinstance(mapped, list) and isinstance(gt, list):
                    loss = sum(loss_fn(m, g) for m, g in zip(mapped, gt, strict=False))
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
                    if cfg.data_loader_kwargs.get("use_val", False) and buf_val is not None:
                        mapper.eval()
                        with th.no_grad():
                            val_loss, n = 0.0, 0
                            for val_act, gt_val in zip(
                                iter(buf_val), iter(gt_val_iter), strict=False
                            ):
                                val_act = val_act.to(cfg.training_device, dtype=cfg.autocast_dtype)
                                gt_val = normalize_gt_batch(gt_val, cfg.training_device)

                                mapped_val = mapper(val_act)
                                if isinstance(mapped_val, tuple):
                                    mapped_val = list(mapped_val)

                                if isinstance(mapped_val, list) and isinstance(gt_val, list):
                                    for m_val, g_val in zip(mapped_val, gt_val, strict=False):
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
    def eval(
        self,
        dataset,
        accessor,
        mapper: th.nn.Module,
        out: Output | None = None,
        model: T2IModel | None = None,
        **kwargs,
    ):
        if self.model is None and model is not None:
            self.model = model
        assert self.model is not None, "Model must be provided at init or in fit()"

        data_loader_kwargs = kwargs.get("data_loader_kwargs", {})
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
            **{
                **data_loader_kwargs,
                "preprocess_fn": data_loader_kwargs.get("gt_processing_fn", None),
                "dataset_column": data_loader_kwargs.get("ground_truth_column", None),
            },
        )
        gt_val_iter = BatchIterator(gt_val_raw, data_loader_kwargs.get("out_batch_size", 1))

        buf_val = _build_buffer(
            gen_val, self.model, accessor, dataset, data_loader_kwargs.get("split", "val"), cfg
        )

        mapper.eval()
        # freeze model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # logging
        # yield Update(info=f"Evaluating KSteer mapper on dataset={dataset} accessor={accessor.attr_name}")

        val_correct = 0
        val_total = 0
        for val_act, gt_val in zip(iter(buf_val), iter(gt_val_iter), strict=False):
            val_act = val_act.to(
                kwargs.get("training_device", "cpu"), dtype=kwargs.get("autocast_dtype", th.float32)
            )
            gt_val = normalize_gt_batch(gt_val, kwargs.get("training_device", "cpu"))

            mapped_val = mapper(val_act)
            if isinstance(mapped_val, tuple):
                mapped_val = list(mapped_val)

            if isinstance(mapped_val, list) and isinstance(gt_val, list):
                for m_val, g_val in zip(mapped_val, gt_val, strict=False):
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
        """Apply gradient-based steering to activations.

        Uses the trained classifier to steer activations toward target attributes
        and away from avoid attributes via gradient descent.

        Args:
            acts: Input activations to steer, shape (batch, features)
            target_idx: Indices of target classes to steer toward (currently uses uniform steering)
            avoid_idx: Indices of classes to avoid (currently uses uniform steering)
            alpha: Step size for gradient updates (default: 1)
            steer_steps: Number of gradient steps (default: 1)
            step_size_decay: Decay factor for alpha per step (default: 1.0, no decay)
            mapper: Mapper/classifier to use. If None, uses self.classifier
            **kwargs: Additional arguments (unused, for compatibility)

        Returns:
            Steered activations of same shape as input

        Raises:
            AssertionError: If classifier not found and mapper not provided

        Note:
            Currently uses `steering_loss_uniform` which balances all classes.
            The target_idx and avoid_idx parameters are not actively used in
            the current implementation but reserved for future targeted steering.

        Example:
            >>> acts = torch.randn(4, 4096*320)  # Batch of 4 activations
            >>> steered = ksteer.steer(acts, alpha=1.5, steer_steps=3)
            >>> steered.shape
            torch.Size([4, 1310720])
        """
        th.set_grad_enabled(True)

        if not hasattr(self, "classifier") and mapper is not None:
            self.classifier = mapper

        assert hasattr(
            self, "classifier"
        ), "Classifier not found. Please fit the model or provide a classifier_path."

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
                    loss_vec = self.steering_loss_uniform(logit[:, -1:])
            else:
                loss_vec = self.steering_loss_uniform(logits[:, -1:])
            loss = loss_vec.mean()
            grads = th.autograd.grad(loss, curr, retain_graph=False)[0]
            current_alpha = alpha * (step_size_decay**step)
            steered = (curr - current_alpha * grads).detach()

        th.set_grad_enabled(False)
        return steered

    # def eval(self, *args,**kwargs) -> None:
    #     pass


class CAA(Steer):
    def __init__(self, model):
        self.model = model

    def fit(self, dataset: dict, accessors: ModuleAccessor, **kwargs):
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
        pos_activations = {accessor.attr_name: [] for accessor in accessors}
        gen = hf_dataset_to_generator(dataset, **{"dataset_column": "positive_prompt"})
        for b in BatchIterator(gen, batch_size):
            with self.model.generate(
                b, num_inference_steps=num_inference_steps, seed=seed, **generate_kwargs
            ) as tracer:
                for accessor in accessors:
                    act = accessor.value
                    pos_activations[accessor.attr_name].append(act.cpu())
                tracer.stop()

        # get negative activations
        neg_activations = {accessor.attr_name: [] for accessor in accessors}
        gen = hf_dataset_to_generator(dataset, **{"dataset_column": "negative_prompt"})
        for b in BatchIterator(gen, batch_size):
            with self.model.generate(
                b, num_inference_steps=num_inference_steps, seed=seed, **generate_kwargs
            ) as tracer:
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

    def eval(self, metric: MetricBase, eval_prompts=list[str], **kwargs):
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

    def steer(self, **kwargs):
        pass


# class ConceptSearch:
#     def __init__(self, model):
#         self.model = model

#     def search_and_steer(self, dataset:dict, accessors:ModuleAccessor, steering_type:Steer, metric:MetricBase, eval_prompts, **kwargs):
#         steering_type.find_directions(dataset, accessors, **kwargs)
#         steering_type.eval(metric, eval_prompts, **kwargs)
#         steering_type.steer(**kwargs)
