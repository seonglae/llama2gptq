import fire

from src.ingest import ingest
from src.qa import chat_cli
from src.quantize import quantization
from constants import (SOURCE_DIRECTORY, PERSIST_DIRECTORY)


def chat(device: str = "cuda") -> str:
  stats = chat_cli(device)
  return stats


def process(src_dir: str = SOURCE_DIRECTORY, dst_dir: str = PERSIST_DIRECTORY, device: str = "cuda") -> str:
  return ingest(src_dir, dst_dir, device)


def quantize(model: str = "ehartford/WizardLM-7B-Uncensored",
             output: str = "wizardlm-7b-uncensored-gptq",
             push: bool = True, owner: str = 'seonglae',
             inference_only: bool = False) -> str:
  quantization(model, output, push, owner, inference_only)
  return 'complete'


if __name__ == '__main__':
  fire.Fire()
