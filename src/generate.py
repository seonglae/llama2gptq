import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, StoppingCriteria, PreTrainedTokenizerBase
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


class TokenStoppingCriteria(StoppingCriteria):
  def __init__(self, target_sequence, prompt, tokenizer: PreTrainedTokenizerBase):
    super().__init__()
    self.target_sequence = target_sequence
    self.prompt = prompt
    self.tokenizer: PreTrainedTokenizerBase = tokenizer

  def __call__(self, input_ids, scores, **kwargs):
    generated_text = self.tokenizer.decode(input_ids[0])
    generated_text = generated_text.replace(self.prompt, "")
    print(generated_text)
    if self.target_sequence in generated_text:
      input_ids[0][-2] = self.tokenizer.bos_token_id
      return True
    return False  # Continue generation

  def __len__(self):
    return 1

  def __iter__(self):
    yield self


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
        use_triton=False,
        use_safetensors=True,
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
