# Angryface

<p align="center">
<img src="img/angryface.png" style="width: 150px"/>
</p>

This project was inspired by the [langchain](https://github.com/hwchase17/langchain) projects like [notion-qa](https://github.com/hwchase17/notion-qa), [localGPT](https://github.com/PromtEngineer/localGPT) and so on.
This project is POC project for [Texomata](https://github.com/texonom/texomata) (browser chat AI). Currently CLO, web UI supports only.

## Install

This project is using [rye](https://mitsuhiko.github.io/rye/) as package manager

```
rye sync

# For cuda support
CUDA_VERSION=cu118
TORCH_VERSION=2.0.1
pip3 install torch==#TORCH_VERSION chromadb --index-url https://download.pytorch.org/whl/$CUDA_VERSION --force
```

## Run
Currently only available with [CUDA](https://texonom.com/a9e934a523d346c5a984d95e3d0676e3)

### Web UI

```zsh
streamlit run chat.py
```

### CLI interaction

```zsh
python main.py chat
```


## Future Plan
- [ ] [MPS](https://texonom.com/8d71e4de36e4416c83f65ee7bdaa412b) support using dynamic model selecting
- [ ] 


## App Stack

### LLM Stack

- [Langchain](https://texonom.com/945567c597364cbb98336ca08c059856) for Prompt Engineering
- [ChromaDB](https://texonom.com/8af886db7d684e03911a86b652620816) for storing embeddings
- [Transformers](https://texonom.com/f5101287cc9249ab812e281e374e5629) for LLM engine
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) for Quantization & Inference

### Python Stack

- [Rye](https://texonom.com/rye-429b5d5f3d7f4026ab5d1abd61facc73) for package management
- [Mypy](https://texonom.com/8a894731430f4138ac0fdd522cd74772) for type checking
- [Fire](https://github.com/google/python-fire) for CLI implementation
- [Streamlit](https://texonom.com/9e295c64d27e4999878a022b1c538964) for Web UI implementation
