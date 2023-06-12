import os
from typing import List, Type, Dict
from pathlib import Path

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.base import BaseLoader
from langchain.vectorstores import Chroma
from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
)

from constants import (CHROMA_SETTINGS)


DOCUMENT_MAP: Dict[str, Type[BaseLoader]] = {
    ".txt": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlxs": UnstructuredExcelLoader,
    ".md": TextLoader,
}


def load_documents(folder_path: str) -> List[Document]:
  glob = Path(f"{folder_path}").glob
  ps = list(glob("**/*.md"))
  documents = []
  for p in ps:
    file_extension = os.path.splitext(p)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
      loader = loader_class(p, encoding="utf-8")
      documents.append(loader.load()[0])
    else:
      continue
  return documents


def ingest(source: str, output: str, device='cuda'):
  print(f"Loading documents from {source}")
  documents = load_documents(source)
  for doc in documents:
    doc.metadata["source"] = str(doc.metadata["source"])
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000, chunk_overlap=200)
  texts = text_splitter.split_documents(documents)
  print(f"Loaded {len(documents)} documents from {source}")
  print(f"Split into {len(texts)} chunks of text")

  embeddings = HuggingFaceInstructEmbeddings(
      model_name="hkunlp/instructor-xl",
      model_kwargs={"device": device},
  )
  db = Chroma.from_documents(
      texts,
      embeddings,
      persist_directory=output,
      client_settings=CHROMA_SETTINGS,
  )
  db.persist()
  db = None
