
import torch
from t2i_interp.t2i import T2IModel
from t2i_interp.accessors.accessor import ModelWrapper
import os

def test_tuple_detection():
    print("Initializing T2IModel for Tuple Detection Verification...")
    # Use standard SD 1.4 which should be cached or available
    model_id = "CompVis/stable-diffusion-v1-4"
    
    # Check if we can run this check (requires internet or cache)
    # We'll try and catch errors
    try:
        # Load on CPU to save GPU memory for this check
        # But 'check_run' inside T2IModel uses 'self.generate'.
        # If the user has a GPU, we might want to use it? 
        # Default t2i.py uses "cpu" if not specified.

        # But let's verify if 't2i_interp/config' files are found.
        if not os.path.exists("t2i_interp/config/modules_to_pick.yaml"):
             print("ERROR: Config files not found in CWD. Run from root.")
             return

        model = T2IModel(model_id, device="cpu", dtype=torch.float32)
    except Exception as e:
        print(f"Skipping verification because model loading failed: {e}")
        # This is expected if no network/cache.
        return

    print("Model loaded. Checking accessors...")
    
    count_tuple = 0
    count_total = 0
    
    sorted_wrappers = sorted(model._wrappers.items())
    
    for wrapper_name, wrapper in sorted_wrappers:
        print(f"--- Wrapper: {wrapper_name} ---")
        accessors = getattr(wrapper, "accessors", {})
        sorted_acc = sorted(accessors.items())
        
        for acc_name, acc in sorted_acc:
            count_total += 1
            if acc.returns_tuple:
                count_tuple += 1
                # print(f"  [TUPLE] {acc_name}")
            else:
                pass
                # print(f"  [TENSOR] {acc_name}")
        
    print(f"\nTotal accessors: {count_total}")
    print(f"Tuple returning: {count_tuple}")
    
    # CLIP encoders often return tuples (last_hidden_state, pooled_output, ...)
    # UNet blocks often return tuples (sample,)
    
    if count_tuple > 0:
        print("SUCCESS: Detected tuple returning accessors.")
    else:
        print("WARNING: No tuple returning accessors detected. This seems suspicious for SD 1.4.")

if __name__ == "__main__":
    test_tuple_detection()
