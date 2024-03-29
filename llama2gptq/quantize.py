import logging
from transformers import AutoTokenizer, TextGenerationPipeline, GenerationConfig, LlamaTokenizer, LlamaTokenizerFast
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


def quantization(source_model: str, output: str, push: bool, owner: str,
                 safetensor=False, inference_only=False):
  logging.basicConfig(
      format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
  )

  tokenizer = AutoTokenizer.from_pretrained(
      source_model, use_fast=True, use_auth_token=True)
  examples = [
      tokenizer(
          "Texonom is an knowledge system that can help you with your daily tasks using AI chatbot."
      )
  ]

  quantize_config = BaseQuantizeConfig(
      bits=4,  # quantize model to 4-bit
      group_size=128,  # it is recommended to set the value to 128
      desc_act=False,  # None act-order can significantly speed up inference but the perplexity may slightly bad
  )

  # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
  if not inference_only:
    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = AutoGPTQForCausalLM.from_pretrained(
        source_model, quantize_config, use_safetensors=safetensor)
    model.quantize(examples)
    model.save_quantized(output, use_safetensors=safetensor)

  # load quantized model to the first GPU
  quantized = AutoGPTQForCausalLM.from_quantized(
      output,
      device="cuda:0",
      use_safetensors=safetensor
  )

  # inference with model.generate
  query = "USER: Are you AI? Say yes or no.\n ASSISTANT:"

  # or you can also use pipeline
  pipeline = TextGenerationPipeline(model=quantized, tokenizer=tokenizer)
  print(pipeline(query)[0]["generated_text"])

  # push quantized model to Hugging Face Hub.
  # to use use_auth_token=True, Login first via huggingface-cli login.
  if push and not inference_only:
    commit_message = f"build: AutoGPTQ for {source_model}" + \
        f": {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
    generation_config = GenerationConfig.from_pretrained(source_model)
    generation_config.push_to_hub(
        output, use_auth_token=True, commit_message=commit_message)
    tokenizer.push_to_hub(output, use_auth_token=True,
                          commit_message=commit_message)
    repo_id = f"{owner}/{output}"
    quantized.push_to_hub(repo_id, use_safetensors=safetensor,
                          commit_message=commit_message, use_auth_token=True)
