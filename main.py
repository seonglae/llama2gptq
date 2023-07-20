import fire

from angryface.ingest import ingest
from angryface.qa import chat_cli
from angryface.quantize import quantization
from constants import (SOURCE_DIRECTORY, PERSIST_DIRECTORY)


def chat(device: str = "cuda") -> str:
  stats = chat_cli(device)
  return stats


def process(src_dir: str = SOURCE_DIRECTORY, dst_dir: str = PERSIST_DIRECTORY, device: str = "cuda") -> str:
  return ingest(src_dir, dst_dir, device)


def quantize(model: str = "meta-llama/Llama-2-13b-chat-hf",
             output: str = "llama-2-13b-chat-hf-gptq",
             push: bool = False, owner: str = 'seonglae',
             safetensor = False, inference_only: bool = False) -> str:
  quantization(model, output, push, owner, safetensor, inference_only)
  return 'complete'


if __name__ == '__main__':
  fire.Fire()
