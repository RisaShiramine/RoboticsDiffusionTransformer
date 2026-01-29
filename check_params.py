
import torch
import torch.nn as nn
from models.rdt.model import RDT

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Default config in model.py
model_default = RDT(
    output_dim=128,
    horizon=32,
    hidden_size=1152,
    depth=28,
    num_heads=16,
    max_lang_cond_len=1024,
    img_cond_len=4096,
)

# 1B config from base.yaml
model_1b = RDT(
    output_dim=128,
    horizon=32,
    hidden_size=2048,
    depth=28,
    num_heads=32,
    max_lang_cond_len=1024,
    img_cond_len=4096,
)

print(f"Default model params (Potential 170M?): {count_parameters(model_default) / 1e6:.2f}M")
print(f"RDT-1B model params: {count_parameters(model_1b) / 1e6:.2f}M")
