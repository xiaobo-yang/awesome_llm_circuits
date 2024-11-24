from typing import List
import torch
import os
import numpy as np
import random




class DataLoaderLite:
    """
        读取fineweb在llama3 tokenizer下的token数据
    """
    def __init__(self, B, T, process_rank, num_processes, split, random_batch=False):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = 'data/edu_fineweb10B/Llama-3'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.random_batch = random_batch  # 新增: 控制是否使用随机batch的标志
        self.reset()

    def reset(self):
        if self.random_batch:
            self.current_shard = random.randint(0, len(self.shards) - 1)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = random.randint(0, len(self.tokens) - self.B * self.T - 1)
        else:
            # 原有的顺序读取逻辑
            self.current_shard = 0
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

    def load_tokens(self, filename):
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def next_batch(self):
        B, T = self.B, self.T
        if self.random_batch:
            while True:
                if self.current_position + B * T + 1 > len(self.tokens):
                    self.current_shard = random.randint(0, len(self.shards) - 1)
                    self.tokens = self.load_tokens(self.shards[self.current_shard])
                    self.current_position = random.randint(0, len(self.tokens) - B * T - 1)
                
                buf = self.tokens[self.current_position : self.current_position + B * T + 1]
                x = buf[:-1].view(B, T)  # inputs
                y = buf[1:].view(B, T)  # targets
                
                self.current_position = random.randint(0, len(self.tokens) - B * T - 1)
                return x, y
        else:
            # 原有的顺序读取逻辑
            buf = self.tokens[self.current_position : self.current_position+B*T+1]
            x = (buf[:-1]).view(B, T) # inputs
            y = (buf[1:]).view(B, T) # targets
            self.current_position += B * T * self.num_processes
            if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = self.load_tokens(self.shards[self.current_shard])
                self.current_position = B * T * self.process_rank
            return x, y




class DataLoaderActivations(DataLoaderLite):
    def __init__(self, model, hook_layers, B, T, process_rank, num_processes, split, random_batch=False, hook_residual=False):
        super().__init__(B, T, process_rank, num_processes, split, random_batch)
        self.model = model
        self.mlp_activations = []
        self.model.eval()

        self.hook_handles = self.hook_mlp(hook_layers, hook_residual)
    
    def next_batch(self):
        x, _ = super().next_batch()
        x = x.to(self.model.device)
        inputs = {'input_ids': x, 'attention_mask': torch.ones_like(x)}
        with torch.no_grad():
            self.model(**inputs)
        return self.mlp_activations.pop()

    def hook_mlp(self, layers: List[int] = [], hook_residual: bool = True):
        def hook_fn(module, input, output):
            output = output.detach().cpu().float().numpy()
            self.mlp_activations.append(output.reshape(-1, output.shape[-1]))
        hook_handles = []
        for layer in layers:
            if hook_residual:
                raise NotImplementedError("residual hook not implemented")
            else:
                hook_handle = self.model.model.layers[layer].mlp.act_fn.register_forward_hook(hook_fn) # 注意这里是llama3架构的命名
            hook_handles.append(hook_handle)
        return hook_handles

    def remove_hook_mlp(self):
        for hook_handle in self.hook_handles:
            hook_handle.remove()
