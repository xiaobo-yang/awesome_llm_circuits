import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import umap
import numpy as np
import tiktoken
import pandas as pd

import sys
sys.path.append("/data/my_tools/build-nanogpt")
from modelling_gpt2 import GPT, GPTConfig


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = '/home/yangxiaobo/my_tools/build-nanogpt/log/gpt2_model_19072.pt'
model_name = model_path.split('/')[-1]
checkpoint = torch.load(model_path, map_location=device, weights_only=True)
model_config = GPTConfig(**checkpoint['config'])
model = GPT(model_config).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()
enc = tiktoken.get_encoding("gpt2")

# 生成正交矩阵
def generate_orthogonal_matrix(dim):
    random_matrix = torch.randn(dim, dim)
    q, r = torch.linalg.qr(random_matrix)
    return q

# 生成768维的正交矩阵
ortho_matrix = generate_orthogonal_matrix(768).to(device)

# 验证正交性（可选）
identity_check = torch.mm(ortho_matrix, ortho_matrix.T)
print(f"正交性检查 - 与单位矩阵的最大误差: {(identity_check - torch.eye(768, device=device)).abs().max().item()}")

# 创建新模型并转换为双精度
new_model = GPT(model_config).to(device)
new_model.load_state_dict(model.state_dict())
new_model.eval()

# 定义LayerNorm的pre和post hook
def ln_pre_hook(module, input):
    x = input[0]
    # 对最后一个维度左乘正交矩阵，反方向旋转
    return (torch.einsum('ij,bkj->bki', ortho_matrix, x), )

def ln_post_hook(module, input, output):
    # 对输出左乘正交矩阵的转置，旋转回来
    return torch.einsum('ij,bkj->bki', ortho_matrix.T, output)

# 注册hook到所有LayerNorm层
with torch.no_grad():
    # 对新模型的每个LayerNorm注册hook
    for layer in range(len(new_model.transformer.h)):
        new_model.transformer.h[layer].ln_1.register_forward_pre_hook(ln_pre_hook)
        new_model.transformer.h[layer].ln_1.register_forward_hook(ln_post_hook)
        new_model.transformer.h[layer].ln_2.register_forward_pre_hook(ln_pre_hook)
        new_model.transformer.h[layer].ln_2.register_forward_hook(ln_post_hook)
    # 最后的ln_f层也需要注册hook
    new_model.transformer.ln_f.register_forward_pre_hook(ln_pre_hook)
    new_model.transformer.ln_f.register_forward_hook(ln_post_hook)

# 对需要旋转的权重进行变换
with torch.no_grad():
    # 1. 词嵌入矩阵 (50304, 768)
    new_model.transformer.wte.weight.data = model.transformer.wte.weight.mm(ortho_matrix)
    # 2. 位置编码矩阵 (1024, 768)
    new_model.transformer.wpe.weight.data = model.transformer.wpe.weight.mm(ortho_matrix)
    # 3. 对每个transformer块中的权重进行旋转
    for layer in range(len(model.transformer.h)):
        # attention权重
        # qkv投影 (768, 2304) -> (768, 768*3)
        new_model.transformer.h[layer].attn.c_attn.weight.data = model.transformer.h[layer].attn.c_attn.weight.data.mm(ortho_matrix)
        new_model.transformer.h[layer].attn.c_proj.weight.data = ortho_matrix.T.mm(model.transformer.h[layer].attn.c_proj.weight.data)
        new_model.transformer.h[layer].attn.c_proj.bias.data = ortho_matrix.T @ model.transformer.h[layer].attn.c_proj.bias.data
        # MLP权重
        # fc1 (768 to 3072)
        new_model.transformer.h[layer].mlp.c_fc.weight.data = model.transformer.h[layer].mlp.c_fc.weight.data.mm(ortho_matrix)
        # fc2 (3072 to 768)
        new_model.transformer.h[layer].mlp.c_proj.weight.data = ortho_matrix.T.mm(model.transformer.h[layer].mlp.c_proj.weight.data)
        new_model.transformer.h[layer].mlp.c_proj.bias.data = ortho_matrix.T @ model.transformer.h[layer].mlp.c_proj.bias.data
    # # 4. 最后的输出层 (50304, 768) # weight tying 故不必操作
    # new_model.lm_head.weight.data = model.lm_head.weight.mm(ortho_matrix)


x = torch.tensor([enc.encode('Hello, this is a test.')]).to(device)
out = model.generate(x, max_new_tokens=32, do_sample=False)[0].tolist()
print(f"original: \n{enc.decode(out)}\n")
rotated_out = new_model.generate(x, max_new_tokens=32, do_sample=False)[0].tolist()
print(f"rotated: \n{enc.decode(rotated_out)}\n")
print(f"Is generation equal: {enc.decode(out) == enc.decode(rotated_out)}")
print(f"Logits difference: {(model(x)[0] - new_model(x)[0]).abs().max().item()}")
