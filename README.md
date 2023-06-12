<p align="center">
<img src="img/angryface.png" style="width: 150px"/>
</p>

# Angryface
This project was inspired by the [langchain](https://github.com/hwchase17/langchain) projects like [notion-qa](https://github.com/hwchase17/notion-qa), [localGPT](https://github.com/PromtEngineer/localGPT) and so on.

## Install
```
rye sync

# For cuda support
CUDA_VERSION=cu118
TORCH_VERSION=2.0.1
pip3 install torch==#TORCH_VERSION chromadb --index-url https://download.pytorch.org/whl/$CUDA_VERSION --force
```
