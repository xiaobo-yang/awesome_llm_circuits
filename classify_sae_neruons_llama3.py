import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import random
import numpy as np
import argparse
import os
import json
from time import time
from datetime import datetime
import wandb
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataloader_activations_llama3 import DataLoaderLite
from sae_model import SAEConfig, SparseAutoencoder
from utils_llama3 import sae_adapt







# ------------------- init ddp -------------------
random_seed = 42
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = random_seed + ddp_rank
torch.manual_seed(seed)


# ------------------- args -------------------
# 训练配置
# sae data args
model_path = '/data/my_data/models/Llama-3.2-1B-Instruct'
sae_checkpoint_path = 'run_20241210_155307_checkpoint_step_119999.pth'
hook_layers = [11,] # layer of mlp to hook
batch_size = 8 # bs太大可能会报RuntimeError: nonzero is not supported for tensors with more than INT_MAX elements, 因为张量中非零元素的数量超过了 INT_MAX（通常是 2^31 - 1）
block_size = 128 # 可能需要和训练时使用的windows size一致
random_batch = True
log_dir = 'log'
num_batches = 100  # 载入的batch个数

class NeuronActivationRecorder:
    def __init__(self, hidden_dim):
        self.batch_info = []
        self.all_infos = defaultdict(list)
    
    def record_hook(self, module, input, output):
        """
            使用hook机制，在model前向传播时记录一个batch内所有token对各neuron的激活值
        """
        mask = output != 0
        indices = mask.nonzero()
        activation_values = output[mask]
        self.batch_info.append((indices, activation_values))
        return output
        
    def process_batch(self, x, model, tokenizer):
        """
            处理一个batch的数据
        """
        with torch.no_grad():
            _ = model(x)  # self.batch_info被更新, 记录了当前batch内所有token对各neuron的激活值
        indices, activation_values = self.batch_info.pop()
        for i, index in tqdm(enumerate(indices), total=len(indices), desc=f"processing one batch"):
            batch_pos, token_pos, neuron_pos = index
            token = tokenizer.decode(x[batch_pos, token_pos])
            context = tokenizer.decode(x[batch_pos, max(0, token_pos-20): token_pos+20])
            self.all_infos[neuron_pos.item()].append(
                (token, context, activation_values[i].item())
            )

# ------------------- load -------------------
# load llm model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, 
    device_map=device  # 自动选择设备（CPU/GPU）
)
model.eval()
# load sae model
sae_ckpt = torch.load(os.path.join(log_dir, sae_checkpoint_path), map_location=device)
autoencoder = SparseAutoencoder(sae_ckpt['config']).to(torch.bfloat16).to(device)
autoencoder.load_state_dict(sae_ckpt['model_state_dict'])
autoencoder.eval()
# hook sae to llm
sae_model = sae_adapt(model, autoencoder, hook_layers)
# load data
data_loader = DataLoaderLite(B=batch_size, T=block_size, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", random_batch=random_batch)
# hook recorder
assert len(hook_layers) == 1, "目前只支持一个层，如果将sae挂在多个层，需要对每个层的neuron激活分别进行记录"
recorder = NeuronActivationRecorder(autoencoder.config.sae_hidden_dim)
record_hook_handle = autoencoder.activation.register_forward_hook(recorder.record_hook)

# ------------------- loop -------------------
for step in range(num_batches):
    x, _ = data_loader.next_batch()
    x = x.to(device)
    recorder.process_batch(x, sae_model, tokenizer)

# test
for i, (neuron_pos, infos) in enumerate(recorder.all_infos.items()):
    infos = sorted(infos, key=lambda x: x[2], reverse=True)
    _, _, activation = infos[0]
    if activation > 1.0:
        for token, context, activation in infos[:10]:
            print(f"[RANK {ddp_rank}] neuron_pos: {neuron_pos}, token: {token}, activation: {activation}\ncontext: {context}\n")
        print('-'*30)

tgt = 'control'
for i, (neuron_pos, infos) in tqdm(enumerate(recorder.all_infos.items()), total=len(recorder.all_infos), desc="find target token"):
    infos = sorted(infos, key=lambda x: x[2], reverse=True)
    token, _, _ = infos[0]
    if tgt in token:
        for token, context, activation in infos[:10]:
            print(f"[RANK {ddp_rank}] neuron_pos: {neuron_pos}, token: {token}, activation: {activation}\ncontext: {context}\n")
        print('-'*30)



# ------------------- clean -------------------
record_hook_handle.remove()

if ddp:
    destroy_process_group()