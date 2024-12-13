import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


from sae_model import SparseAutoencoder
from utils_llama3 import sae_adapt

model_path = '/data/my_data/models/Llama-3.2-1B-Instruct'
sae_checkpoint_path = 'run_20241210_163535_checkpoint_step_79999.pth'
hook_layers = [11,] # layer of mlp to hook
batch_size = 8 # bs太大可能会报RuntimeError: nonzero is not supported for tensors with more than INT_MAX elements, 因为张量中非零元素的数量超过了 INT_MAX（通常是 2^31 - 1）
block_size = 1024 # 可能需要和训练时使用的windows size一致
random_batch = True
log_dir = 'log'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# load llm model ----------------------------
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




# activate neuron ----------------------------
act_neurons = [24978, 28899, 54133, 51204, 59532] # america
gen_prompt = 'Vladimir Putin is the president of'
inputs = tokenizer(gen_prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
gen_inputs = inputs['input_ids']
topk = 5 # 输出前topk个概率最大next token及其概率
output_file_path = 'modify_activations_output.txt'  # 指定输出文件路径
with open(output_file_path, 'w') as f:
    for val in range(1,300):
        val /= 2.5
        # activated generation
        def act_neuron(module, input, output):
            for neuron in act_neurons:
                output[:, :, neuron] = val
            return output
        with torch.no_grad():
            hook_handle = autoencoder.activation.register_forward_hook(act_neuron)
            slg = sae_model(**inputs).logits
            sae_output = sae_model.generate(**inputs, max_new_tokens=50, do_sample=False)
            sae_text = tokenizer.batch_decode(sae_output, skip_special_tokens=False)
            hook_handle.remove()
            spb = slg.softmax(dim=-1)
        with torch.no_grad():
            hook_handle = autoencoder.activation.register_forward_hook(act_neuron)
            slg_all = sae_model(sae_output).logits
            slg = slg_all[0, gen_inputs.shape[1]-1]
            slg2 = slg_all[0, gen_inputs.shape[1]]
            slg3 = slg_all[0, gen_inputs.shape[1]+1]
            hook_handle.remove()
            # hook_handle0.remove()
            spb = slg.softmax(dim=-1)
            spb2 = slg2.softmax(dim=-1)
            spb3 = slg3.softmax(dim=-1)
        sva, sid = spb.topk(topk)
        sva2, sid2 = spb2.topk(topk)
        sva3, sid3 = spb3.topk(topk)
        token1 = tokenizer.decode([sae_output[0, gen_inputs.shape[1]-1]])
        token2 = tokenizer.decode([sae_output[0, gen_inputs.shape[1]]])
        token3 = tokenizer.decode([sae_output[0, gen_inputs.shape[1]+1]])
        print(f"SAE activation value: {val}\n    The first token for generation: '{token1}'\n       next token: {sid.tolist()}\n       next token prob: {sva}\n       next token decoded: {[tokenizer.decode([token]) for token in sid.tolist()]}\n    The second token for generation: '{token2}'\n       next token: {sid2.tolist()}\n       next token prob: {sva2}\n       next token decoded: {[tokenizer.decode([token]) for token in sid2.tolist()]}\n    The third token for generation: '{token3}'\n       next token: {sid3.tolist()}\n       next token prob: {sva3}\n       next token decoded: {[tokenizer.decode([token]) for token in sid3.tolist()]}")
        print(f"SAE-Model: {sae_text}\n")
        f.write(f"SAE activation value: {val}\n    The first token for generation: '{token1}'\n       next token: {sid.tolist()}\n       next token prob: {sva}\n       next token decoded: {[tokenizer.decode([token]) for token in sid.tolist()]}\n    The second token for generation: '{token2}'\n       next token: {sid2.tolist()}\n       next token prob: {sva2}\n       next token decoded: {[tokenizer.decode([token]) for token in sid2.tolist()]}\n    The third token for generation: '{token3}'\n       next token: {sid3.tolist()}\n       next token prob: {sva3}\n       next token decoded: {[tokenizer.decode([token]) for token in sid3.tolist()]}\n")
        f.write(f"SAE-Model: {sae_text}\n\n")
    with torch.no_grad():
        lg = model(**inputs).logits
        pb = lg.softmax(dim=-1)
        va, id = pb.topk(topk)
    orig_output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    orig_text = tokenizer.batch_decode(orig_output, skip_special_tokens=False)
    print(f'Model:\n    prob: {va}\n    position: {id}\n')
    print(f"Model: {orig_text}")
    f.write(f'Model:\n    prob: {va}\n    position: {id}\n')
    f.write(f"Model: {orig_text}\n")
