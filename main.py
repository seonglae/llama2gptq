import fire

from src.ingest import ingest
from src.qa import qa_loop
from constants import (SOURCE_DIRECTORY, PERSIST_DIRECTORY)


def chat(device: str = "cuda") -> str:
  stats = qa_loop(device)
  return stats


def process(src_dir: str = SOURCE_DIRECTORY, dst_dir: str = PERSIST_DIRECTORY, device: str = "cuda") -> str:
  return ingest(src_dir, dst_dir, device)


if __name__ == '__main__':
  fire.Fire()
