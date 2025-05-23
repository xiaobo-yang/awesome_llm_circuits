import copy
from dataclasses import dataclass
import torch
import torch.nn as nn
import os



@dataclass
class SAEConfig:
    # 稀疏自编码器配置
    sae_input_dim: int = 8192  # 修改为 GPT2 的嵌入维度
    sae_hidden_dim: int = 131072  # 隐藏层维度,大于输入维度
    sae_l1_coefficient: float = 5.0  # L1正则化系数
    sae_l2_norm: float = 0.1  # encoder weight初始化的正则化权重范围



class SparseAutoencoder(nn.Module):
    """
        ref: https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_dim, hidden_dim = config.sae_input_dim, config.sae_hidden_dim
        
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.activation = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
        self.lambda_coef = config.sae_l1_coefficient  # This can be adjusted as needed
        self.l2_norm = config.sae_l2_norm
        
        self.initialize_weights(self.l2_norm)

    def initialize_weights(self, l2_norm):
        # l2 norm推荐范围为0.05-1，0.1是最常用的值
        with torch.no_grad():
            for col in range(self.decoder.weight.size(1)):
                col_vector = torch.randn(self.decoder.weight.size(0))
                col_vector *= l2_norm / torch.norm(col_vector)
                self.decoder.weight[:, col] = col_vector
        
        self.encoder.weight = nn.Parameter(self.decoder.weight.T) 
        # https://transformer-circuits.pub/2024/april-update/index.html#training-saes 这里只提到初始化满足tying，训练时不tied。不过后续可以自己实验看看。
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        return self.activation(self.encoder(x))

    def decode(self, f):
        return self.decoder(f)

    def forward(self, x):
        x = preprocess_dataset(x) # TODO: 这个看上去不太好被接入原始model中：生成时需要也对embedding做标准化，且产生一个token都有重新标准化，不太方便。。
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    def loss_fn(self, x, x_hat, f):
        # Reconstruction loss
        mse_loss = ((x - x_hat)**2).sum(dim=1).mean()
        
        # Sparsity loss: 
        l2_norm_W_d = self.decoder.weight.norm(dim=0)
        sparsity_loss = self.lambda_coef * (torch.abs(f) * l2_norm_W_d).sum(dim=1).mean()
        
        total_loss = mse_loss + sparsity_loss
        return total_loss, mse_loss, sparsity_loss

def preprocess_dataset(X):
    n = X.size(-1)
    current_norm = X.norm(dim=-1).mean() # 一个seq中每个token的embedding向量，2-norm平均值
    target_norm = torch.sqrt(torch.tensor(n).float())
    scaling_factor = target_norm / current_norm
    X_scaled = X * scaling_factor
    return X_scaled

# TODO: 以下fsdp的分片脚本未完善，先写在这里占位。搞清楚fsdp原理后再修改。。
if __name__ == '__main__':
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        CPUOffload,
        MixedPrecision,
        ShardingStrategy,
        BackwardPrefetch,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        size_based_auto_wrap_policy,
        enable_wrap,
        wrap,
    )
    
    # 1. 初始化分布式环境
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    
    # 2. 在CPU上创建模型
    if local_rank == 0:
        config = SAEConfig(
            sae_input_dim=32,
        sae_hidden_dim=64,
        )
        model = SparseAutoencoder(config)  # 不要调用.to(device)
    else:
        model = None
    dist.barrier()
    
    # 3. 配置FSDP
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    fsdp_config = dict(
        device_id=local_rank,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全切分
        mixed_precision=mp_policy,
        cpu_offload=CPUOffload(offload_params=True),  # CPU卸载
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # 预取优化
        limit_all_gathers=True,  # 限制all-gather操作
    )
    
    # 4. 使用FSDP包装模型并广播
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model = FSDP(model, **fsdp_config)
    model = model.cuda(local_rank)
    
    # 5. 打印模型信息（仅在rank 0上）
    if local_rank == 0:
        print(f"Model initialized with FSDP:")
        print(f"- Total parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"- FSDP config: {fsdp_config}")
    
    # 6. 清理
    dist.destroy_process_group()

