# LLaMa2 GPTQ

Chat AI which can provide responses with reference documents by Prompt engineering over vector database. It suggests related web pages provided through the integration with my previous product, Texonom.

<p align="center">
<img src="img/angryface.png" style="width: 150px"/>
</p>

Pursuing local, private and personal AI without requesting external API attained by optimizing inference performance with GPTQ model quantization. This project was inspired by the [langchain](https://github.com/hwchase17/langchain) projects like [notion-qa](https://github.com/hwchase17/notion-qa), [localGPT](https://github.com/PromtEngineer/localGPT).

# Demos

### CLI Demo

https://github.com/seonglae/llama2gptq/assets/27716524/dba5cd39-ea5c-44d9-bf29-2e8f04039413

### Chat Demo

https://github.com/seonglae/llama2gptq/assets/27716524/258de629-0b61-4670-b76b-9f2357adf4c7

<br/>

## Install

This project is using [rye](https://mitsuhiko.github.io/rye/) as package manager
Currently only available with [CUDA](https://texonom.com/a9e934a523d346c5a984d95e3d0676e3)

```
rye sync
```

or using pip

```
CUDA_VERSION=cu118
TORCH_VERSION=2.0.1
pip install torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/$CUDA_VERSION --force
pip install torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/$CUDA_VERSION
pip install .
```

## QA

### 1. Chat with Web UI

```zsh
streamlit run chat.py
```

### 2. Chat with CLI

```zsh
python main.py chat
```

## Ingest Documents

Currently code structure is mainly focussed on Notion's csv exported data

### Custom source documents

```zsh
# Put document files to ./knowledge folder
python main.py ingest
# Or use provided Texonom DB
git clone https://huggingface.co/datasets/texonom/texonom-md db
```

## Quantize Model

Default model is orca 3b for now

```zsh
python main quantize --source_model facebook/opt-125m --output opt-125m-4bit-gptq --push
```

## Future Plan

- [ ] [MPS](https://texonom.com/8d71e4de36e4416c83f65ee7bdaa412b) support using dynamic model selecting
- [ ] Stateful Web App support like [chat-langchain](https://chat.langchain.dev/)

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
