# Angryface


## Install
```
rye sync

# For cuda support
CUDA_VERSION=cu118
TORCH_VERSION=2.0.1
pip3 install torch==#TORCH_VERSION chromadb --index-url https://download.pytorch.org/whl/$CUDA_VERSION --force
```
