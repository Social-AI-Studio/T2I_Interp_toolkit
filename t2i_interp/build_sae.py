import torch
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from t2i_interp.sae import SAEManager

def build_sae_manager(model, saes_config, device="cuda:0", dtype=torch.float16):
    """
    Builds the SAEManager from a dictionary of SAE configurations.
    
    Args:
        model: T2IModel instance.
        saes_config: dict or OmegaConf mapping, e.g.:
            {
                "model_unet_down_blocks_2_attentions_1_out": {
                    "path": "./checkpoints/...",
                    "k": 10,
                    "hidden_dim": 5120
                },
                ...
            }
        device: targeted device string.
        dtype: torch dtype.
    
    Returns:
        sae_manager: SAEManager instance tracking the SAEs.
        sae_list: list of tuples to pass directly into inference functions:
                  [(model_module_ref, sae_instance, "hook_name"), ...]
    """
    sae_list = []
    
    for hook_name, config in saes_config.items():
        # Load the SAE from the given path
        # Note: AutoEncoderTopK.from_pretrained expects a path to state_dict.pth
        sae = AutoEncoderTopK.from_pretrained(
            config.path,
            k=config.k,
            device=device,
        ).to(dtype)
        
        # Override scalar k to tensor so dictionary_learning internals don't break
        hidden_dim = config.get("hidden_dim", 5120)
        sae.k = torch.Tensor([hidden_dim]).to(device, dtype=dtype)
        
        # We need to map the string "model_unet_down_blocks_2_attentions_1_out" 
        # to the actual module reference on our `model` object.
        # String replacement logic based on how T2IModel stores wrappers:
        # e.g., "model_unet_down_blocks_2_attentions_1_out" -> model.unet.down_blocks_2_attentions_1_out
        attr_path = hook_name.replace("model_", "") # strip leading base prefix if present
        
        # Safely traverse the model object to find the module slice
        module_ref = model
        for part in attr_path.split("_"):
            # A bit tricky since attributes might be `down_blocks_2_attentions_1_out` straight on `unet`
            # or nested. In `T2IModel`, the wrapped names are exactly the hook names minus "model_".
            pass 
            
        # Due to how `T2IModel` constructs proxy hooks, the module reference is usually 
        # just an attribute lookup on `model.unet` matching the exact hook name minus "model_unet_".
        if hook_name.startswith("model_unet_"):
            attr_name = hook_name[11:] # remove "model_unet_"
            module_ref = getattr(model.unet, attr_name)
        else:
            module_ref = getattr(model, hook_name)
            
        sae_list.append((module_ref, sae, hook_name))

    sae_manager = SAEManager(model=model)
    return sae_manager, sae_list
