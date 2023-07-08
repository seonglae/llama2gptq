import logging
from transformers import LlamaTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model_dir = "psmathur/orca_mini_3b"
quantized_model_dir = "orca-mini-3b-4bit-gptq"

tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
)

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalLM.from_pretrained(
    pretrained_model_dir, quantize_config)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples)
model.save_quantized(quantized_model_dir)
model.save_quantized(quantized_model_dir, use_safetensors=True)

# load quantized model to the first GPU
model = AutoGPTQForCausalLM.from_quantized(
    quantized_model_dir, device="cuda:0")

# inference with model.generate
query = "Angryface is "
print(tokenizer.decode(model.generate(
    **tokenizer(query, return_tensors="pt").to(model.device))[0]))

# or you can also use pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline(query)[0]["generated_text"])


# push quantized model to Hugging Face Hub.
# to use use_auth_token=True, Login first via huggingface-cli login.
# or pass explcit token with: use_auth_token="hf_xxxxxxx"
repo_id = f"seonglae/{quantized_model_dir}"
commit_message = f"AutoGPTQ model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
model.push_to_hub(repo_id, save_dir=quantized_model_dir, use_safetensors=True, commit_message=commit_message, use_auth_token=True)
