from enum import Enum
import torch as th
import re
import yaml
from loguru import logger

class IOType(Enum):
    """Enum to specify input or output access"""
    INPUT = "input"
    OUTPUT = "output"

class ModuleAccessor:
    """
    I/O accessor that acts as a handle to a specific module and access point (input/output).
    Used by T2IModel and DiffusionIntervention to identify where to hook.

    Future Goal: Standardize naming conventions for various T2I models (e.g., SD1.5, SDXL, PixArt)
    to provide a unified interface for interpretation research, similar to `nnterp`.
    """

    def __init__(
        self,
        module: th.nn.Module,
        attr_name: str | None,
        io_type: IOType,
    ):
        self.module = module
        self.attr_name = attr_name
        self.io_type = io_type

    @property
    def heads(self) -> int | None:
        """
        Returns number of attention heads if available on the module.
        Useful for attention-based interventions.
        """
        return getattr(self.module, "heads", None)
    
    def __repr__(self):
        return f"<ModuleAccessor {self.attr_name} ({self.io_type.value})>"


class ModelWrapper:
    """
    Generic container for model accessors, dynamically created based on a configuration file.
    
    Iterates through the model's named modules and creates input/output accessors
    for any module whose name matches the patterns specified in the config.
    """

    def __init__(self, module: th.nn.Module, config_path: str):
        self.module = module
        self.accessors = {}
        
        # Load patterns from config
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                patterns = config.get('patterns', [])
        except FileNotFoundError:
            # Fallback or empty if config not found
            patterns = []
            # logger.warning(f"Config file not found at {config_path}. No dynamic accessors will be created.")

        self._create_dynamic_accessors(patterns)

    def _create_dynamic_accessors(self, patterns: list[str]):
        """
        Walks the module tree and creates accessors for matching modules.
        """
        for name, module in self.module.named_modules():
            # Check if module name matches any pattern
            for pattern in patterns:
                if re.search(pattern, name):
                    # Create a sanitized attribute name from the module path
                    # e.g., "down_blocks.0.resnets.0" -> "down_blocks_0_resnets_0"
                    sanitized_name = name.replace(".", "_")
                    
                    # Create Input Accessor
                    input_accessor_name = f"{sanitized_name}_in"
                    if not hasattr(self, input_accessor_name):
                        input_accessor = ModuleAccessor(module, input_accessor_name, IOType.INPUT)
                        setattr(self, input_accessor_name, input_accessor)
                        self.accessors[input_accessor_name] = input_accessor
                    
                    # Create Output Accessor
                    output_accessor_name = f"{sanitized_name}_out"
                    if not hasattr(self, output_accessor_name):
                        output_accessor = ModuleAccessor(module, output_accessor_name, IOType.OUTPUT)
                        setattr(self, output_accessor_name, output_accessor)
                        self.accessors[output_accessor_name] = output_accessor
                    
                    # Break after first match to avoid duplicate accessors for the same module 
                    # if it matches multiple patterns (optional behavior)
                    break

    def summary(self) -> str:
        s = f"{self.module.__class__.__name__} Dynamic Accessors:\n"
        if not self.accessors:
            s += "  No accessors created.\n"
        else:
            # Sort by name for readable output
            for name in sorted(self.accessors.keys()):
                s += f"  - {name}\n"
        return s

    def __getattr__(self, name):
        return getattr(self.module, name)

    def __repr__(self):
        return self.summary()
