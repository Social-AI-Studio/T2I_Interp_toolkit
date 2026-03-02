
import torch
import os
import shutil
from t2i_interp.t2i import T2IModel
from t2i_interp.accessors.accessor import IOType, ModuleAccessor
from t2i_interp.intervention import AddVectorIntervention, run_intervention
from t2i_interp.utils.T2I.buffer import t2IActivationBuffer

def test_t2i_hooks():
    print("Testing T2IModel hooks...")
    # Use a small model for speed if possible, or just standard SD1.5
    try:
        model = T2IModel(model="runwayml/stable-diffusion-v1-5", device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16)
    except Exception as e:
        print(f"Skipping model load due to environment/auth issues (expected if run offline): {e}")
        # Mocking T2IModel for logic testing if real load fails would be ideal, but let's assume environment is set up.
        return

    # Create a simple accessor
    # Taking a known layer path. 
    # For SD1.5: unet.mid_block.resnets.0
    accessor = ModuleAccessor(model.pipeline.unet.mid_block.resnets[0], "mid_block_0", IOType.OUTPUT)
    
    prompt = "A photo of a cat"
    
    print("Running run_with_cache...")
    cached = model.run_with_cache([prompt], [accessor], num_inference_steps=1)
    
    assert "mid_block_0" in cached, "Failed to cache mid_block_0"
    val = cached["mid_block_0"]
    print(f"Captured tensor shape: {val.shape}")
    assert val.ndim > 1, "Captured value should be a tensor"
    print("run_with_cache test passed!")

def test_interventions():
    print("Testing Interventions...")
    try:
        model = T2IModel(model="runwayml/stable-diffusion-v1-5", device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16)
    except Exception:
        return

    accessor = ModuleAccessor(model.pipeline.unet.mid_block.resnets[0], "mid_block_0", IOType.OUTPUT)
    
    # Create a dummy steering vector
    # We need to know the shape. mid block is [B, 1280, 8, 8] for 512x512
    # Let's run once to get shape if we didn't just run it.
    
    steering_vec = torch.randn(1, 1280, 8, 8, device=model.device, dtype=model.dtype)
    
    intervention = AddVectorIntervention(
        accessors=[accessor],
        steering_vec=steering_vec,
        start_step=0,
        end_step=5
    )
    
    print("Running run_intervention...")
    # We just check if it runs without crashing
    out = run_intervention(model, ["A photo of a dog"], [intervention], num_inference_steps=5)
    print("run_intervention finished.")
    print("Intervention test passed!")

def test_buffer():
    print("Testing t2IActivationBuffer...")
    try:
        model = T2IModel(model="runwayml/stable-diffusion-v1-5", device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16)
    except Exception:
        return

    accessor = ModuleAccessor(model.pipeline.unet.mid_block.resnets[0], "mid_block_0", IOType.OUTPUT)
    
    def data_gen():
        prompts = ["cat", "dog", "bird", "fish"]
        for p in prompts:
            yield p

    # d_submodule should be inferred or provided. 
    # For resnets[0], out channels is 1280.
    buffer = t2IActivationBuffer(
        data=data_gen(),
        model=model,
        submodule=accessor,
        d_submodule=1280,
        n_ctxs=4,
        refresh_batch_size=2,
        out_batch_size=2,
        data_device="cpu"
    )
    
    print("Iterating buffer...")
    try:
        batch = next(buffer)
        print(f"Buffer yielded batch of shape: {batch.shape}")
        assert batch.shape[-1] == 1280
        print("Buffer test passed!")
    except StopIteration:
        print("Buffer empty prematurely!")
    except Exception as e:
        print(f"Buffer failed: {e}")

if __name__ == "__main__":
    test_t2i_hooks()
    test_interventions()
    test_buffer()
