import torch
import copy

def sae_adapt(model, autoencoder, layers):
    """
        add sae adapter to model
        Input:
            layer: layer with sae
    """
    sae_model = copy.deepcopy(model)
    
    def sae_hook(module, input, output):
        # 在gate_proj和up_proj的乘积之后应用autoencoder
        encoded = autoencoder.encoder(output)
        activated = autoencoder.activation(encoded)
        decoded = autoencoder.decoder(activated)
        return decoded
    
    for layer in layers:
        sae_model.model.layers[layer].mlp.act_fn.register_forward_hook(sae_hook) # 注意：只适合llama3的module name（act_fn）
    
    return sae_model

def compare_gen(model, tokenizer, autoencoder, layers):
    """
        compare sae adapter with original model
        Input:
            layer: layer with sae
    """
    prompts = [
        "The quick brown fox",
        "In a world where",
        "Once upon a time",
        "The future of AI"
    ]
    device = model.device
    sae_model = sae_adapt(model, autoencoder, layers) # deepcopy不改变device
    inputs = tokenizer(prompts, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    sae_output = sae_model.generate(**inputs, max_new_tokens=50, do_sample=False)
    orig_output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    sae_texts = tokenizer.batch_decode(sae_output, skip_special_tokens=False)
    orig_texts = tokenizer.batch_decode(orig_output, skip_special_tokens=False)
    samples = [
        {"prompt": prompt, "sae_model": sae_text, "model": orig_text}
        for prompt, sae_text, orig_text in zip(prompts, sae_texts, orig_texts)
    ]
    for sample in samples:
        print(f"SAE-model: {sample['sae_model']}")
        print(f"model: {sample['model']}")
    return samples
