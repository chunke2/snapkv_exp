# 测试一下模型的 head 配置
import torch
from transformers import AutoModelForCausalLM

MODEL_PATH = "/workspace/models/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="cuda")
attn = model.model.layers[0].self_attn
print(f"num_heads: {attn.num_heads}")
print(f"num_key_value_heads: {attn.num_key_value_heads}")
print(f"num_key_value_groups: {attn.num_key_value_groups}")
