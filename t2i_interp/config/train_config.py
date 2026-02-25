import itertools
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import torch as t

from dictionary_learning.dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    JumpReluAutoEncoder,
)
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE, BatchTopKTrainer
from dictionary_learning.trainers.gdm import GatedSAETrainer
from dictionary_learning.trainers.jumprelu import JumpReluTrainer
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
    MatryoshkaBatchTopKTrainer,
)
from dictionary_learning.trainers.p_anneal import PAnnealTrainer
from dictionary_learning.trainers.standard import StandardTrainer, StandardTrainerAprilUpdate
from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKTrainer

# from dictionary_learning.trainers.sampledsae import (
#     SampledActivationTrainer,
#     SampledActivationSAE,
#     HybridSampledTopKTrainer,
#     HybridSampledTopKSAE,
# )


DEBUG = False


class TrainerType(Enum):
    STANDARD = "standard"
    STANDARD_NEW = "standard_new"
    TOP_K = "top_k"
    BATCH_TOP_K = "batch_top_k"
    GATED = "gated"
    P_ANNEAL = "p_anneal"
    JUMP_RELU = "jump_relu"
    Matryoshka_BATCH_TOP_K = "matryoshka_batch_top_k"
    # SAMPLED_SAE = "sampled_sae"
    # HYBRID_SAMPLED_TOP_K = "hybrid_sampled_top_k"


@dataclass
class LLMConfig:
    llm_batch_size: int
    context_length: int
    sae_batch_size: int
    dtype: t.dtype


@dataclass
class SparsityPenalties:
    standard: list[float]
    standard_new: list[float]
    p_anneal: list[float]
    gated: list[float]


num_tokens = 25_000_000  # 500 million tokens


print(f"NOTE: Training on {num_tokens} tokens")

eval_num_inputs = 200
random_seeds = [0, 1, 2]
dictionary_widths = [2**16]

WARMUP_STEPS = 1000
SPARSITY_WARMUP_STEPS = 5000
DECAY_START_FRACTION = 0.8

learning_rates = [3e-4]

# wandb_project = "seeded-sampled-sae-project"

# LLM_CONFIG = {
#     "EleutherAI/pythia-70m-deduped": LLMConfig(
#         batch_size=64, context_length=77, sae_batch_size=2048, dtype=t.float32
#     ), #original batch size 2048,
# }

SPARSITY_PENALTIES = SparsityPenalties(
    standard=[0.012, 0.015, 0.02, 0.03, 0.04, 0.06],
    standard_new=[0.012, 0.015, 0.02, 0.03, 0.04, 0.06],
    p_anneal=[0.006, 0.008, 0.01, 0.015, 0.02, 0.025],
    gated=[0.012, 0.018, 0.024, 0.04, 0.06, 0.08],
)


TARGET_L0s = [60]


@dataclass
class BaseTrainerConfig:
    activation_dim: int
    device: str
    layer: str
    lm_name: str
    submodule_name: str
    trainer: type[Any]
    dict_class: type[Any]
    wandb_name: str
    warmup_steps: int
    steps: int
    decay_start: int | None


@dataclass
class StandardTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: int | None
    resample_steps: int | None = None


@dataclass
class StandardNewTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: int | None


@dataclass
class PAnnealTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    initial_sparsity_penalty: float
    sparsity_warmup_steps: int | None
    sparsity_function: str = "Lp^p"
    p_start: float = 1.0
    p_end: float = 0.2
    anneal_start: int = 10000
    anneal_end: int | None = None
    sparsity_queue_length: int = 10
    n_sparsity_updates: int = 10


@dataclass
class TopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000  # when to begin tracking the average threshold


# @dataclass
# class SampledActivationTrainerConfig(BaseTrainerConfig):
#     dict_size: int
#     seed: int
#     lr: float
#     k: int
#     sampling_update_freq: int = 1
#     sampling_method: str = "leverage"
#     ridge_lambda: float = 0.01
#     sketching_size: Optional[int] = None
#     auxk_alpha: float = 1 / 32
#     threshold_beta: float = 0.999
#     threshold_start_step: int = 1000


# @dataclass
# class HybridSampledTopKTrainerConfig(BaseTrainerConfig):
#     dict_size: int
#     seed: int
#     lr: float
#     k: int
#     l_multiplier: float = 1.5
#     sampling_update_freq: int = 1
#     sampling_method: str = "leverage"
#     ridge_lambda: float = 0.01
#     sketching_size: Optional[int] = None
#     auxk_alpha: float = 1 / 32
#     threshold_beta: float = 0.999
#     threshold_start_step: int = 1000


@dataclass
class MatryoshkaBatchTopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    group_fractions: list[float] = field(
        default_factory=lambda: [
            (1 / 32),
            (1 / 16),
            (1 / 8),
            (1 / 4),
            ((1 / 2) + (1 / 32)),
        ]
    )
    group_weights: list[float] | None = None
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000  # when to begin tracking the average threshold


@dataclass
class GatedTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: int | None


@dataclass
class JumpReluTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    target_l0: int
    sparsity_warmup_steps: int | None
    sparsity_penalty: float = 1.0
    bandwidth: float = 0.001


def sae_trainer_config(
    architecture: str,
    learning_rates: list[float],
    seeds: list[int],
    activation_dim: int,
    dict_sizes: list[int],
    model_name: str,
    device: str,
    layer: str,
    submodule_name: str,
    steps: int,
    warmup_steps: int = WARMUP_STEPS,
    sparsity_warmup_steps: int = SPARSITY_WARMUP_STEPS,
    decay_start_fraction=DECAY_START_FRACTION,
) -> list[dict]:
    decay_start = int(steps * decay_start_fraction)

    trainer_configs = []

    base_config = {
        "activation_dim": activation_dim,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "decay_start": decay_start,
        "device": device,
        "layer": layer,
        "lm_name": model_name,
        "submodule_name": submodule_name,
    }
    if TrainerType.P_ANNEAL.value == architecture:
        for seed, dict_size, learning_rate, sparsity_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.p_anneal
        ):
            config = PAnnealTrainerConfig(
                **base_config,
                trainer=PAnnealTrainer,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                initial_sparsity_penalty=sparsity_penalty,
                wandb_name=f"PAnnealTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.STANDARD.value == architecture:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.standard
        ):
            config = StandardTrainerConfig(
                **base_config,
                trainer=StandardTrainer,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                wandb_name=f"StandardTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.STANDARD_NEW.value == architecture:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.standard_new
        ):
            config = StandardNewTrainerConfig(
                **base_config,
                trainer=StandardTrainerAprilUpdate,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                wandb_name=f"StandardTrainerNew-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.GATED.value == architecture:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.gated
        ):
            config = GatedTrainerConfig(
                **base_config,
                trainer=GatedSAETrainer,
                dict_class=GatedAutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                wandb_name=f"GatedTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.TOP_K.value == architecture:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = TopKTrainerConfig(
                **base_config,
                trainer=TopKTrainer,
                dict_class=AutoEncoderTopK,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                wandb_name=f"TopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.BATCH_TOP_K.value == architecture:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = TopKTrainerConfig(
                **base_config,
                trainer=BatchTopKTrainer,
                dict_class=BatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                wandb_name=f"BatchTopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.Matryoshka_BATCH_TOP_K.value == architecture:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = MatryoshkaBatchTopKTrainerConfig(
                **base_config,
                trainer=MatryoshkaBatchTopKTrainer,
                dict_class=MatryoshkaBatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                wandb_name=f"MatryoshkaBatchTopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.JUMP_RELU.value == architecture:
        for seed, dict_size, learning_rate, target_l0 in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = JumpReluTrainerConfig(
                **base_config,
                trainer=JumpReluTrainer,
                dict_class=JumpReluAutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                target_l0=target_l0,
                wandb_name=f"JumpReluTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    # if TrainerType.SAMPLED_SAE.value == architecture:
    #     # Define available sampling methods to experiment with
    #     #sampling_methods = ["leverage", "entropy", "uniform", "l2_norm", "coreset"]
    #     sampling_methods = ["leverage", "entropy", "uniform", "l2_norm"]
    #     for seed, dict_size, learning_rate, k, method in itertools.product(
    #         seeds, dict_sizes, learning_rates, TARGET_L0s, sampling_methods
    #     ):
    #         # Set appropriate ridge_lambda based on the method
    #         ridge_lambda = 0.01
    #         if method == "coreset":
    #             ridge_lambda = 0.01  # Default value for coreset

    #         # Set appropriate sketching_size for coreset method
    #         sketching_size = None
    #         if method == "coreset":
    #             sketching_size = min(dict_size // 10, 100)

    #         config = SampledActivationTrainerConfig(
    #             **base_config,
    #             trainer=SampledActivationTrainer,
    #             dict_class=SampledActivationSAE,
    #             lr=learning_rate,
    #             dict_size=dict_size,
    #             seed=seed,
    #             k=k,
    #             sampling_method=method,
    #             ridge_lambda=ridge_lambda,
    #             sketching_size=sketching_size,
    #             wandb_name=f"SampledActivationTrainer-{method}-{model_name}-{submodule_name}",
    #         )
    #         trainer_configs.append(asdict(config))

    # if TrainerType.HYBRID_SAMPLED_TOP_K.value == architecture:
    #     sampling_methods = ["leverage", "entropy", "uniform", "l2_norm"]
    #     l_multipliers = [3.0]
    #     for seed, dict_size, learning_rate, k, method, l_mult in itertools.product(
    #         seeds, dict_sizes, learning_rates, TARGET_L0s, sampling_methods, l_multipliers
    #     ):
    #         sketching_size = None
    #         if method == "coreset":
    #             sketching_size = min(dict_size // 10, 100)

    #         config = HybridSampledTopKTrainerConfig(
    #             **base_config,
    #             trainer=HybridSampledTopKTrainer,
    #             dict_class=HybridSampledTopKSAE,
    #             lr=learning_rate,
    #             dict_size=dict_size,
    #             seed=seed,
    #             k=k,
    #             l_multiplier=l_mult,
    #             sampling_method=method,
    #             ridge_lambda=0.01,
    #             sketching_size=sketching_size,
    #             wandb_name=f"HybridSampledTopKTrainer-{method}-{model_name}-{submodule_name}",
    #         )
    #         trainer_configs.append(asdict(config))
    return trainer_configs
