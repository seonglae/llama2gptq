from time import time
from typing import Tuple, List

import torch
from transformers import Pipeline


from src.generate import load_embeddings, load_db, load_model, TokenStoppingCriteria


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
  print(query)

  # Inference
  criteria = TokenStoppingCriteria(user_token, query, transformer.tokenizer)
  response = transformer(query, stopping_criteria=criteria)[
      0]["generated_text"]
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
    pingpong = qa_cli(device, db, transformer, history)
    if len(pingpong) == 0:
      break
    pingongs.append(pingpong)
  return pingongs
