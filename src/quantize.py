import logging
from transformers import AutoTokenizer, TextGenerationPipeline, GenerationConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


def quantization(source_model: str, output: str, push: bool, owner: str, inference_only: bool = True):
  logging.basicConfig(
      format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
  )

  tokenizer = AutoTokenizer.from_pretrained(source_model, use_fast=True)
  examples = [
      tokenizer(
          "Angryface is an AI assistant that can help you with your daily tasks."
      )
  ]

  quantize_config = BaseQuantizeConfig(
      bits=4,  # quantize model to 4-bit
      group_size=128,  # it is recommended to set the value to 128
      desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
  )

  # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
  if not inference_only:
    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = AutoGPTQForCausalLM.from_pretrained(
        source_model, quantize_config)
    model.quantize(examples)
    model.save_quantized(output)
    model.save_quantized(output, use_safetensors=True)

  # load quantized model to the first GPU
  model = AutoGPTQForCausalLM.from_quantized(
      output,
      device="cuda:0"
  )

  # inference with model.generate
  query = "USER: Are you AI? Say yes or no.\n ASSISTANT:"

  # or you can also use pipeline
  pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
  print(pipeline(query)[0]["generated_text"])

  # push quantized model to Hugging Face Hub.
  # to use use_auth_token=True, Login first via huggingface-cli login.
  # or pass explcit token with: use_auth_token="hf_xxxxxxx"
  if push and not inference_only:
    commit_message = f"build: AutoGPTQ for {source_model}" + \
        f": {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
    generation_config = GenerationConfig.from_pretrained(source_model)
    generation_config.push_to_hub(
        output, use_auth_token=True, commit_message=commit_message)
    repo_id = f"{owner}/{output}"
    tokenizer.push_to_hub(output, save_dir=output,
                          use_auth_token=True, commit_message=commit_message)
    model.push_to_hub(repo_id, save_dir=output, use_safetensors=True,
                      commit_message=commit_message, use_auth_token=True)
