import torch

from quant_matmul.quantize import replace_linear_weight_only_int8, replace_linear_weight_only_int4


from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.models.llama import llama_config_to_gpt2_config, remap_state_dict_hf_llama
from flash_attn.utils.pretrained import state_dict_from_pretrained
from flash_attn.utils.benchmark import pytorch_profiler

device = "cuda"
dtype = torch.float16

model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "meta-llama/Llama-2-13b-hf"
config = llama_config_to_gpt2_config(
    AutoConfig.from_pretrained(model_name, trust_remote_code=True)
)
config.use_flash_attn = True
config.fused_dropout_add_ln = True
config.residual_in_fp32 = True
pretrained_state_dict = remap_state_dict_hf_llama(state_dict_from_pretrained(model_name), config)
model = GPTLMHeadModel(config, device=device, dtype=dtype)
model.load_state_dict(pretrained_state_dict)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt").to(device)
max_length = input_ids.shape[-1] + 100

out = model.generate(
    input_ids=input_ids,
    max_length=max_length,
    cg=True,
    return_dict_in_generate=True,
    output_scores=True,
    enable_timing=True,
)
print(tokenizer.batch_decode(out.sequences.tolist()))
# pytorch_profiler(model.generate, input_ids, max_length=max_length, cg=True, trace_filename="llama_7b_generation_fp16.json")
del model._decoding_cache

replace_linear_weight_only_int8(model.transformer)
out = model.generate(
    input_ids=input_ids,
    max_length=max_length,
    cg=True,
    return_dict_in_generate=True,
    output_scores=True,
    enable_timing=True,
)
print(tokenizer.batch_decode(out.sequences.tolist()))
# pytorch_profiler(model.generate, input_ids, max_length=max_length, cg=True, trace_filename="llama_7b_generation_int8.json")
del model._decoding_cache

model = GPTLMHeadModel(config, device=device, dtype=dtype)
model.load_state_dict(pretrained_state_dict)
model.eval()
replace_linear_weight_only_int4(model.transformer)
out = model.generate(
    input_ids=input_ids,
    max_length=max_length,
    cg=True,
    return_dict_in_generate=True,
    output_scores=True,
    enable_timing=True,
)
print(tokenizer.batch_decode(out.sequences.tolist()))
# pytorch_profiler(model.generate, input_ids, max_length=max_length, cg=True, trace_filename="llama_7b_generation_int4.json")
