from time import time
from typing import Tuple, List

import torch
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY


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
  return db


def load_model(
    device: str, model_id="TheBloke/WizardLM-7B-uncensored-GPTQ",
    model_basename="WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order",
    model_type="gptq",
):
  assert device == "cuda"
  tokenizer = AutoTokenizer.from_pretrained(
      model_id, use_fast=True)

  if model_type == "gptq":
    model = AutoGPTQForCausalLM.from_quantized(
        model_id,
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        device='cuda:0',
        use_triton=False
    )
  elif model_type == "auto":
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='cuda:0',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.tie_weights()

  pipe = pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
      max_length=2048,
      temperature=0.01,
      top_p=0.95,
      repetition_penalty=1.5,
  )
  llm = HuggingFacePipeline(pipeline=pipe)
  return llm


def qa(query, device, db, embeddings, llm, history: List[List[str]]) -> Tuple:
  print(f"Running on: {device}")
  if embeddings is None:
    embeddings = load_embeddings(device)
  if db is None:
    db = load_db(embeddings)
  if llm is None:
    llm = load_model(device)

  # input similarity
  start = time()
  input_refs = db.search(query, search_type="similarity")
  for document in input_refs:
    print("\n> " + document.metadata["source"])
  print(f"Time taken: {time() - start} seconds\n")

  # Inference
  start = time()
  retriever = Chroma.from_documents(input_refs, embeddings).as_retriever()
  chain = RetrievalQA.from_chain_type(
      llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True,
      callbacks=[StreamingStdOutCallbackHandler()],
  )

  # History Prompt
  prompt = [f"Question: {q}\nAnswer: {a}\n" for [q, a] in history]
  query = "".join(prompt) + f'Question: {query}\nAnswer: '
  print(query)
  res = chain(query)
  answer, answer_refs = res["result"], res["source_documents"]
  for document in answer_refs:
    print("\n> " + document.metadata["source"])
  print(f"Time taken: {time() - start} seconds\n")

  # output similarity
  start = time()
  output_refs = db.search(answer, search_type="similarity")

  # Print the result
  for document in output_refs:
    print("\n> " + document.metadata["source"])
  print(f"Time taken: {time() - start} seconds\n")
  print(query + answer + '\n')
  return (input_refs, answer_refs, answer, output_refs)


def qa_cli(device, db, embeddings, llm, history) -> Tuple:
  query = input("\nQuestion: ")
  if query == "exit":
    return ()
  return (query, *qa(query, device, db, embeddings, llm, history))


def chat_cli(device='cuda'):
  embeddings = load_embeddings(device)
  db = load_db(embeddings)
  llm = load_model(device)

  pingongs = []
  while True:
    history = [[pingpong[0], pingpong[3]] for pingpong in pingongs]
    pingongs.append(qa_cli(device, db, embeddings, llm, history))
  return pingongs
