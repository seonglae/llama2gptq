from time import time
from typing import Tuple, List

import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, Pipeline
from auto_gptq import AutoGPTQForCausalLM

from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY


def load_embeddings(device):
  embeddings = HuggingFaceInstructEmbeddings(
      model_name="hkunlp/instructor-xl", model_kwargs={"device": device}
  )
  return embeddings


def load_db(device, embeddings=None):
  if embeddings is None:
    embeddings = load_embeddings(device)
  db = Chroma(
      persist_directory=PERSIST_DIRECTORY,
      embedding_function=embeddings,
      client_settings=CHROMA_SETTINGS,
  )
  return db


def load_model(
    device: str, model_id="seonglae/wizardlm-7b-uncensored-gptq",
    model_basename="gptq_model-4bit-128g",
    model_type="gptq",
):
  assert device == "cuda"

  if model_type == "llama":
    tokenizer = LlamaTokenizer.from_pretrained(
        model_id, use_fast=True)
  else:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True)

  if model_type == "gptq":
    model = AutoGPTQForCausalLM.from_quantized(
        model_id,
        model_basename=model_basename,
        trust_remote_code=True,
        device='cuda:0',
        use_triton=False
    )
  elif model_type == "llama":
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        device_map='cuda:0',
        torch_dtype=torch.float16,
        trust_remote_code=True,
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
  model.eval()

  transformer = pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
      temperature=0.5,
      top_p=0.95,
      max_new_tokens=100,
      repetition_penalty=1.15,
  )
  return transformer


@torch.no_grad()
def qa(query, device, db, transformer: Pipeline, history: List[List[str]],
       user_token="USER: ",
       bot_token="ASSISTANT: ",
       sys_token="",
       system="") -> Tuple:
  start = time()

  if db is None:
    embeddings = load_embeddings(device)
    db = load_db(device, embeddings)
  if transformer is None:
    transformer = load_model(device)

  # input similarity
  prompt = [f"{user_token}{q}\n{bot_token}{a}\n" for [q, a] in history]
  query = f"{sys_token}{system}" + \
      "".join(prompt) + f'{user_token}{query}\n{bot_token}'
  print("Conversation Refs\n")
  print(query)

  # Inference
  response = transformer(query)[0]["generated_text"]
  answer = response.replace(query, "").strip()

  # output similarity
  answer_refs = db.search(query + answer, search_type="similarity")

  # Print the result
  for document in answer_refs:
    print("\n> " + document.metadata["source"])
  print("Answer Refs\n")
  print(f"Time taken: {time() - start} seconds\n")
  print(query + answer + '\n')

  return (answer, answer_refs)


def qa_cli(device, db, llm, history) -> Tuple:
  query = input("\nQuestion: ")
  if query == "exit":
    return ()
  return (query, *qa(query, device, db, llm, history))


def chat_cli(device='cuda'):
  embeddings = load_embeddings(device)
  db = load_db(device, embeddings)
  transformer = load_model(device)

  pingongs = []
  while True:
    history = [[pingpong[0], pingpong[1]] for pingpong in pingongs]
    pingongs.append(qa_cli(device, db, transformer, history))
  return pingongs
