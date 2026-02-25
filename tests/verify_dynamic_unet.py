
import torch
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextConfig, T5EncoderModel, T5Config
from t2i_interp.accessors.unet import Unet
from t2i_interp.accessors.clip_encoder import ClipEncoder
from t2i_interp.accessors.t5_encoder import T5Encoder
import os

def test_dynamic_accessors():
    print("Testing Dynamic Accessors (UNet, CLIP, T5)...")
    
    # --- UNet Verification ---
    print("\n--- Testing UNet ---")
    unet = UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32,
    )
    unet_config = "t2i_interp/config/unet.yaml"
    if os.path.exists(unet_config):
        print(f"Initializing Unet wrapper with config: {unet_config}")
        unet_wrapper = Unet(unet, config_path=unet_config)
        found_resnet = any("resnets" in k for k in unet_wrapper.accessors.keys())
        found_attn = any("attn" in k for k in unet_wrapper.accessors.keys())
        print(f"UNet Check: Resnet={found_resnet}, Attn={found_attn}")
    else:
        print(f"ERROR: Config not found {unet_config}")

    # --- CLIP Verification ---
    print("\n--- Testing CLIP ---")
    clip_config = CLIPTextConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=256,
        projection_dim=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=77
    )
    clip_model = CLIPTextModel(clip_config)
    clip_yaml = "t2i_interp/config/CLIPEncoder.yaml"
    if os.path.exists(clip_yaml):
        print(f"Initializing ClipEncoder wrapper with config: {clip_yaml}")
        clip_wrapper = ClipEncoder(clip_model, config_path=clip_yaml)
        # Check for layers matching: layers, mlp, self_attn, layer_norm, encoder
        found_mlp = any("mlp" in k for k in clip_wrapper.accessors.keys())
        found_attn = any("self_attn" in k for k in clip_wrapper.accessors.keys())
        print(f"CLIP Check: MLP={found_mlp}, SelfAttn={found_attn}")
        print("Sample keys:", list(clip_wrapper.accessors.keys())[:3])
    else:
         print(f"ERROR: Config not found {clip_yaml}")

    # --- T5 Verification ---
    print("\n--- Testing T5 ---")
    t5_config = T5Config(
        vocab_size=1000,
        d_model=64,
        d_kv=64,
        d_ff=256,
        num_layers=2,
        num_heads=4,
        decoder_start_token_id=0,
        is_encoder_decoder=False # Encoder only
    )
    t5_model = T5EncoderModel(t5_config)
    t5_yaml = "t2i_interp/config/T5EncoderModel.yaml"
    if os.path.exists(t5_yaml):
        print(f"Initializing T5Encoder wrapper with config: {t5_yaml}")
        t5_wrapper = T5Encoder(t5_model, config_path=t5_yaml)
        # Check for: block, layer, SelfAttention, DenseReluDense, final_layer_norm
        found_block = any("block" in k for k in t5_wrapper.accessors.keys())
        found_dense = any("DenseReluDense" in k for k in t5_wrapper.accessors.keys())
        print(f"T5 Check: Block={found_block}, Dense={found_dense}")
        print("Sample keys:", list(t5_wrapper.accessors.keys())[:3])
    else:
        print(f"ERROR: Config not found {t5_yaml}")

if __name__ == "__main__":
    test_dynamic_accessors()
