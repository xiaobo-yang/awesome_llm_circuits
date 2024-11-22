"""
    生成fineweb数据集在llama3 tokenizer下的token数据
    
    ref: https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py
"""


import os
import multiprocessing as mp
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = 'data/edu_fineweb10B/Llama-3'
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("/data/my_data/datasets/fineweb-edu", split="train", num_proc=52)

# init the tokenizer
model_path = "/data/my_data/models/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
eot = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(tokenizer(doc["text"])['input_ids'])
    tokens_np = np.array(tokens)
    tokens_np_uint32 = tokens_np.astype(np.uint32) # llama3 token最大到128256，不能使用uint16（最大65536）
    return tokens_np_uint32

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint32)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
