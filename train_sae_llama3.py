import torch
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
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataloader_activations_llama3 import DataLoaderActivations
from sae_model import SAEConfig, SparseAutoencoder
from utils_llama3 import compare_gen




# 添加命令行参数解析
parser = argparse.ArgumentParser(description="训练或恢复稀疏自编码器训练")
parser.add_argument('--hook_layers', '-l', type=int, default=[11,], help='模型挂载sae的层')
parser.add_argument('--resume', '-r', action='store_true', help='从sae检查点恢复训练')
parser.add_argument('--checkpoint', '-c', type=str, default='', help='sae检查点文件路径')
parser.add_argument('--not_wandb', action='store_true', help='不使用Wandb进行日志记录')
args = parser.parse_args()


# 设置分布式训练
random_seed = 42
is_distri = int(os.environ.get('RANK', -1)) != -1
if is_distri:
    init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    master_process = rank == 0
else:
    rank = 0
    local_rank = 0
    world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = random_seed + rank
torch.manual_seed(seed)


# 训练配置
# sae data args
model_path = '/data/my_data/models/Llama-3.2-1B-Instruct'
hook_layers = args.hook_layers # layer of mlp to hook
batch_size = 1
total_batch_size = batch_size * 8  # this is bs of token, bs of mlp activations is total_batch_size * block_size
grad_accum_steps = max(1, total_batch_size // (batch_size * world_size))
block_size = 128
random_batch = False
# sae training args
num_steps = 200000
ini_lr = 1e-5
clip_norm = 1.0
ini_lambda = 5.0 # lambda for sparsity loss
sae_l2_norm = 0.1  # initialized norm of encoder
save_steps = 10000
eval_interval = 100
num_eval_steps = 40 // world_size # eval sample num = num_eval_steps * batch_size

def get_lr(step):
    threshold = int(0.8*num_steps)
    return ini_lr if step < threshold else ini_lr * (num_steps - step)/(num_steps - threshold)

def get_lambda(step):
    threshold = int(0.05*num_steps)
    return ini_lambda if step >= threshold else ini_lambda * step / threshold


# ------------------- load -------------------
# load llm model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, 
    device_map=device  # 自动选择设备（CPU/GPU）
)
model.eval()
# load data
data_loader = DataLoaderActivations(model, hook_layers=hook_layers, B=batch_size, T=block_size, process_rank=rank, num_processes=world_size, split="train", random_batch=random_batch)
# load sae model
config = SAEConfig(
    sae_input_dim=8192,
    sae_hidden_dim=131072,
    sae_l1_coefficient=ini_lambda,
    sae_l2_norm=sae_l2_norm,
)
autoencoder = SparseAutoencoder(config).to(torch.bfloat16).to(device)
autoencoder.eval()

# wandb
if master_process:
    wandb_config = {
        "model": model_path.split("/")[-1],
        "hook_layers": hook_layers,
        "batch_size": batch_size,
        "block_size": block_size,
        "random_batch": random_batch,
        "ini_lr": ini_lr,
        "clip_norm": clip_norm,
        "ini_lambda": ini_lambda,
        "sae_l2_norm": sae_l2_norm,
        "random_seed": random_seed,
    }
    wandb_config.update(config.__dict__)
    print(f"Autoencoder input dim: {config.sae_input_dim}")
    print(f"Autoencoder hidden dim: {config.sae_hidden_dim}")
# resume
if args.resume and os.path.exists(args.checkpoint):
    checkpoint = torch.load(args.checkpoint)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(autoencoder.parameters(), lr=ini_lr, betas=(0.9, 0.999))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    wandb_id = checkpoint['wandb_id']
    start_step = checkpoint['step']
    if master_process:
        print(f"step {start_step}恢复训练")
        if not args.not_wandb:
            wandb.init(project="sae-llama3", id=wandb_id, resume="must")
            run_name = wandb.run.name
        else:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{current_time}"
else:
    optimizer = optim.Adam(autoencoder.parameters(), lr=ini_lr, betas=(0.9, 0.999))
    start_step = 0
    if master_process:
        print("从头开始训练自编码器")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not args.not_wandb:
            wandb.init(project="sae-llama3", name=f"run_{current_time}", config=wandb_config)
            run_name = wandb.run.name
        else:
            run_name = f"run_{current_time}"
# log
if master_process:
    if not os.path.exists('log'):
        os.makedirs('log', exist_ok=True)
    json_filename = f"log/{run_name}_samples.json"
   


# ------------------- train -------------------
if is_distri:
    autoencoder = DDP(autoencoder)
raw_autoencoder = autoencoder.module if is_distri else autoencoder

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

for step in range(start_step, num_steps):
    if master_process:
        start_time = time()

    # ------------------- train -------------------
    autoencoder.train()
    for param_group in optimizer.param_groups:
        param_group['lr'] = get_lr(step)
    raw_autoencoder.lambda_coef = get_lambda(step)

    loss_accum, recon_loss_accum, sparsity_loss_accum, l1_loss_accum, l0_loss_accum = 0.0, 0.0, 0.0, 0.0, 0.0
    for micro_step in range(grad_accum_steps):
        batch = data_loader.next_batch()
        batch = torch.from_numpy(batch).to(torch.bfloat16).to(device)
        if is_distri:
            autoencoder.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        reconstructed, encoded = autoencoder(batch)
        loss, recon_loss, sparsity_loss = raw_autoencoder.loss_fn(batch, reconstructed, encoded)
        loss /= grad_accum_steps
        loss.backward()

        loss_accum += loss.detach()
        recon_loss_accum += recon_loss.detach() / grad_accum_steps
        sparsity_loss_accum += sparsity_loss.detach() / grad_accum_steps
        with torch.no_grad():
            l1_loss = encoded.abs().sum(dim=1).mean()
            l0_loss = (encoded != 0).float().sum(dim=1).mean()
            l1_loss_accum += l1_loss.detach() / grad_accum_steps
            l0_loss_accum += l0_loss.detach() / grad_accum_steps
    if is_distri:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        dist.all_reduce(recon_loss_accum, op=dist.ReduceOp.AVG)
        dist.all_reduce(sparsity_loss_accum, op=dist.ReduceOp.AVG)
        dist.all_reduce(l1_loss_accum, op=dist.ReduceOp.AVG)
        dist.all_reduce(l0_loss_accum, op=dist.ReduceOp.AVG)
    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), clip_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


    # 记录损失到wandb
    if master_process:
        print(f"step {step} total_loss: {loss_accum.item():.4f}, recon_loss: {recon_loss_accum.item():.4f}, sparsity_loss: {sparsity_loss_accum.item():.4f}, l1_loss: {l1_loss_accum.item():.4f}, l0_loss: {l0_loss_accum.item():.4f}")
        if not args.not_wandb:
            wandb.log({
                "step": step,
                "total_loss": loss_accum.item(),
                "reconstruction_loss": recon_loss_accum.item(),
                "sparsity_loss": sparsity_loss_accum.item(),
                "lr": optimizer.param_groups[0]['lr'],
                "lambda": raw_autoencoder.lambda_coef,
                "l1_loss": l1_loss_accum.item(),
                "l0_loss": l0_loss_accum.item(),
            })
    # ------------------- eval -------------------
    if (step + 1) % eval_interval == 0:
        autoencoder.eval()

        eval_losses = []
        eval_recon_losses = []
        eval_sparsity_losses = []
        eval_l1_losses = []
        eval_l0_losses = []
        with torch.no_grad():
            for eval_step in range(num_eval_steps):
                eval_batch = data_loader.next_batch()
                eval_batch = torch.from_numpy(eval_batch).to(torch.bfloat16).to(device)

                eval_reconstructed, eval_encoded = autoencoder(eval_batch)
                eval_loss, eval_recon_loss, eval_sparsity_loss = raw_autoencoder.loss_fn(eval_batch, eval_reconstructed, eval_encoded)
                eval_l1_loss = eval_encoded.abs().sum(dim=1).mean()
                eval_l0_loss = (eval_encoded != 0).float().sum(dim=1).mean()
                
                eval_losses.append(eval_loss.item())
                eval_recon_losses.append(eval_recon_loss.item())
                eval_sparsity_losses.append(eval_sparsity_loss.item())
                eval_l1_losses.append(eval_l1_loss.item())
                eval_l0_losses.append(eval_l0_loss.item())
        
        eval_losses = torch.tensor(eval_losses, device=device)
        eval_recon_losses = torch.tensor(eval_recon_losses, device=device)
        eval_sparsity_losses = torch.tensor(eval_sparsity_losses, device=device)
        eval_l1_losses = torch.tensor(eval_l1_losses, device=device)
        eval_l0_losses = torch.tensor(eval_l0_losses, device=device)
        if is_distri:
            dist.all_reduce(eval_losses, op=dist.ReduceOp.AVG)
            dist.all_reduce(eval_recon_losses, op=dist.ReduceOp.AVG)
            dist.all_reduce(eval_sparsity_losses, op=dist.ReduceOp.AVG)
            dist.all_reduce(eval_l1_losses, op=dist.ReduceOp.AVG)
            dist.all_reduce(eval_l0_losses, op=dist.ReduceOp.AVG)

        avg_eval_loss = eval_losses.mean().item()
        avg_eval_recon_loss = eval_recon_losses.mean().item()
        avg_eval_sparsity_loss = eval_sparsity_losses.mean().item()
        avg_eval_l1_loss = eval_l1_losses.mean().item()
        avg_eval_l0_loss = eval_l0_losses.mean().item()

        if master_process:
            print(f"  Eval Total Loss: {avg_eval_loss:.4f}")
            print(f"  Eval Reconstruction Loss: {avg_eval_recon_loss:.4f}")
            print(f"  Eval Sparsity Loss: {avg_eval_sparsity_loss:.4f}")
            print(f"  Eval L1 Loss: {avg_eval_l1_loss:.4f}")
            print(f"  Eval L0 Loss: {avg_eval_l0_loss:.4f}")
        
            # 比较生成样本并保存
            samples = compare_gen(model, tokenizer, raw_autoencoder, hook_layers)
            with open(json_filename, 'a') as f:
                json.dump(samples, f, indent=2)
        
            # 记录评估结果到wandb
            if not args.not_wandb:
                wandb.log({
                    "eval_total_loss": avg_eval_loss,
                    "eval_reconstruction_loss": avg_eval_recon_loss,
                    "eval_sparsity_loss": avg_eval_sparsity_loss,
                    "eval_l1_loss": avg_eval_l1_loss,
                    "eval_l0_loss": avg_eval_l0_loss,
                    "generation_samples": wandb.Table(
                            columns=["Prompt", "model with SAE", "model"],
                            data=[[s["prompt"], s["sae_model"], s["model"]] for s in samples]
                        )
                })
            

    # ------------------- save model -------------------
    if master_process and ((step + 1) % save_steps == 0 or step == num_steps - 1):
        checkpoint_path = f"log/{run_name}_checkpoint_step_{step}.pth"
        checkpoint_data = {
            'step': step + 1,
            'config': config,
            'model_state_dict': raw_autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if not args.not_wandb:
            checkpoint_data['wandb_id'] = wandb.run.id
        torch.save(checkpoint_data, checkpoint_path)
        print(f"save to {checkpoint_path}")
    
    if master_process:
        end_time = time()
        print(f"step {step} time: {end_time - start_time:.2f}s")




# ------------------- clean -------------------
data_loader.remove_hook_mlp()

if is_distri:
    destroy_process_group()
