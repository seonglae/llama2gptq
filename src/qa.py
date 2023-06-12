from time import time
from typing import Tuple

import torch
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY


def load_model(device: str):
  model_id = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ"
  model_basename = "Wizard-Vicuna-7B-Uncensored-GPTQ-4bit-128g.no-act-order"
  tokenizer = AutoTokenizer.from_pretrained(
      model_id, use_fast=True, max_length=4096)
  model = AutoGPTQForCausalLM.from_quantized(
      model_id,
      model_basename=model_basename,
      use_safetensors=True,
      trust_remote_code=True,
      device='cuda:0',
      use_triton=False,
      quantize_config=None)

  pipe = pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
      max_length=4096,
      temperature=0,
      top_p=0.95,
      repetition_penalty=1,
  )
  llm = HuggingFacePipeline(pipeline=pipe)
  return llm


def load_embeddings(device):
  embeddings = HuggingFaceInstructEmbeddings(
      model_name="hkunlp/instructor-xl", model_kwargs={"device": device}
  )
  return embeddings


def load_db(embeddings):
  db = Chroma(
      persist_directory=PERSIST_DIRECTORY,
      embedding_function=embeddings,
      client_settings=CHROMA_SETTINGS,
  )
  print(f"{db._client.list_collections()[0].count()} documents loaded")
  return db


def ask(query, device, db, embeddings, llm) -> Tuple:
  print(f"Running on: {device}")
  if embeddings is None:
    embeddings = load_embeddings(device)
  if db is None:
    db = load_db(embeddings)
  if llm is None:
    llm = load_model(device)

  # input similarity
  start = time()
  input_refs = db.similarity_search(query, search_type="similarity")
  print(f"Time taken: {time() - start} seconds")
  for document in input_refs:
    print("\n> " + document.metadata["source"])

  # inference
  start = time()
  retriever = Chroma.from_documents(input_refs, embeddings).as_retriever()
  qa = RetrievalQA.from_chain_type(
      llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True,
      callbacks=[StreamingStdOutCallbackHandler()]
  )
  res = qa(query)
  answer, answer_refs = res["result"], res["source_documents"]
  print(f"Time taken: {time() - start} seconds")

  # output similarity
  start = time()
  output_refs = db.search(answer, search_type="similarity")
  print(f"Time taken: {time() - start} seconds")

  # Print the result
  print("\n\n> Question:")
  print(query)
  print("\n> Answer:")
  print(answer)
  return (input_refs, answer_refs, answer, output_refs)


def qa(device, db, embeddings, llm) -> Tuple:
  query = input("\nQuestion: ")
  if query == "exit":
    return ()
  return (query, *ask(query, device, db, embeddings, llm))


def qa_loop(device='cuda'):
  embeddings = load_embeddings(device)
  db = load_db(embeddings)
  llm = load_model(device)

  conversation = []
  while True:
    conversation.append(qa(device, db, embeddings, llm))
  return conversation
