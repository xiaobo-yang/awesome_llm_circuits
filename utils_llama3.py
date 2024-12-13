import torch
import copy

def sae_adapt(model, autoencoder, layers, hook_residual=True):
    """
        add sae adapter to model
        Input:
            layer: layer with sae
    """
    sae_model = copy.deepcopy(model)
    
    for layer in layers:
        if hook_residual:
            layer = sae_model.model.layers[layer]
            layer.sae = autoencoder  # 添加sae属性
            layer.forward = custom_forward.__get__(layer)  # 绑定新的forward方法
        else:
            def sae_hook(module, input, output):
                # 在gate_proj和up_proj的乘积之后应用autoencoder
                encoded = autoencoder.encoder(output)
                activated = autoencoder.activation(encoded)
                decoded = autoencoder.decoder(activated)
                return decoded
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


# customed llama3 block layer forward
def custom_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: None,
    position_ids: None,
    past_key_value: None,
    output_attentions: False,
    use_cache: False,
    cache_position: None,
    position_embeddings: None,
    **kwargs,
):
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    # 应用SAE到residual connection
    if hasattr(self, 'sae'):
        encoded = self.sae.encoder(residual)
        activated = self.sae.activation(encoded)
        residual = self.sae.decoder(activated)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs