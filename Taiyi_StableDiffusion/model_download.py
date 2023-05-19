import os

import torch
from diffusers import StableDiffusionPipeline

# proxy
os.environ["http_proxy"] = "http://100.72.64.19:12798"
os.environ["https_proxy"] = "http://100.72.64.19:12798"

# mkdir
os.makedirs("/root/autodl-tmp/models", exist_ok=True)
os.makedirs("/root/autodl-tmp/outputs", exist_ok=True)

# download model
model_ids = ["IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1",
             "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"]
for model_id in model_ids:
    StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

print("Successful Download All model cache")