# Installation Guide - T2I Interpretability Toolkit

## Quick Install

### 1. Install Python Requirements

```bash
cd /home/ubuntu/T2I_Interp_toolkit
pip install -r requirements.txt
```

### 2. Install the Toolkit (Development Mode)

```bash
# Install the main toolkit
cd /home/ubuntu/T2I_Interp_toolkit
pip install -e .

# Install the dictionary learning submodule
cd dictionary_learning
pip install -e .
```

## Alternative: Install with Poetry

If you prefer using Poetry for the dictionary learning module:

```bash
cd /home/ubuntu/T2I_Interp_toolkit/dictionary_learning
poetry install
```

## GPU Support

Make sure you have PyTorch with CUDA support:

```bash
# Check your CUDA version
nvcc --version

# Install PyTorch with CUDA (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## For FairFace Dataset

If you plan to use the FairFace dataset features:

```bash
# Install dlib (may require system dependencies)
pip install dlib

# On Ubuntu, if dlib fails, install dependencies first:
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
sudo apt-get install libx11-dev libgtk-3-dev
pip install dlib
```

## Verify Installation

```python
# Test basic imports
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test toolkit imports
from t2Interp.T2I import T2IModel
from t2Interp.intervention import EncoderAttentionIntervention
print("✅ Toolkit imported successfully!")

# Test diffusers
from diffusers import AutoPipelineForText2Image
print("✅ Diffusers imported successfully!")
```

## Common Issues

### Issue: "No module named 't2Interp'"
**Solution:** Make sure you're in the notebook directory or the toolkit is installed:
```bash
cd /home/ubuntu/T2I_Interp_toolkit/notebooks
# Run your notebook from here
```

### Issue: dlib installation fails
**Solution:** Install system dependencies:
```bash
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev
pip install dlib
```

### Issue: CUDA out of memory
**Solution:** Reduce batch size or use mixed precision:
```python
# In your code
pipe = pipe.to("cuda", torch_dtype=torch.float16)
```

## Environment Setup

For a clean installation, consider using a virtual environment:

```bash
# Create virtual environment
python -m venv t2i_env

# Activate it
source t2i_env/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Minimal Requirements

If you just want to run the debiasing notebook:

```bash
pip install torch torchvision diffusers transformers pandas scikit-learn numpy matplotlib
```

## Development Installation

For development with all optional dependencies:

```bash
pip install -r requirements.txt
pip install pytest black flake8 mypy  # Development tools
```


