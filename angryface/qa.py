from time import time
from typing import Tuple, List

import torch
from transformers import Pipeline

from angryface.ingest import extract_ref
from angryface.generate import load_embeddings, load_db, load_model, TokenStoppingCriteria


@torch.no_grad()
def qa(query, device, db, transformer: Pipeline, history: List[List[str]],
       user_token="USER: ",
       bot_token="ASSISTANT: ",
       sys_token="",
       system="",
       extract_ref=extract_ref) -> Tuple:
  start = time()

  if db is None:
    embeddings = load_embeddings(device)
    db = load_db(device, embeddings)
  if transformer is None:
    transformer = load_model(device)

  # input similarity
  conversation = [f"{user_token}{q}\n{bot_token}{a}\n" for [q, a] in history]
  prompt = f"{sys_token}{system}" + \
      "".join(conversation) + f'{user_token}{query}\n{bot_token}'
  print(prompt)

  # Inference
  criteria = TokenStoppingCriteria(
      user_token.strip(), prompt, transformer.tokenizer)
  response = transformer(prompt, stopping_criteria=criteria)[
      0]["generated_text"]
  answer = response.replace(prompt, "").strip()

  # output similarity
  refs = db.similarity_search_with_relevance_scores(
      f'{user_token}{query}\n{bot_token}', search_type="similarity")
  print([ref[1] for ref in refs])
  refs = [ref[0] for ref in refs]

  # Print the result
  print('\nHelpful links\n')
  for ref in refs:
    ref_info = extract_ref(ref)
    print(f"{ref_info['title']}: {ref_info['link']}")

  print(f"\nTime taken: {time() - start} seconds\n")
  print(prompt + answer + '\n')

  return (answer, refs)


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
