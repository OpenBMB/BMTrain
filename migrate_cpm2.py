from collections import OrderedDict
import torch
from tqdm import tqdm

def main():
    inp = torch.load("/root/ckpts/cpm2.1/merge.pt")
    out = OrderedDict()
    out["input_embedding.weight"] = inp["word_embeds.weight"]
    out["output_projection.weight"] = inp["lm_head.weight"]
    out["layernorm_after_enc.weight"] = inp["encoder.final_layernorm.weight"]
    out["position_bias_enc.weight"] = inp["encoder.blocks.0.self_attn.self_attn.relative_attention_bias.weight"].transpose(0, 1).contiguous()
    out["layernorm_after_dec.weight"] = inp["decoder.final_layernorm.weight"]
    out["position_bias_dec.weight"] = inp["decoder.blocks.0.self_attn.self_attn.relative_attention_bias.weight"].transpose(0, 1).contiguous()
    for i in tqdm(range(24), desc="Encoder"):
        prefix = f"enc_layers.{i}"
        old_prefix = f"encoder.blocks.{i}"
        out[f"{prefix}.layernorm_before_attention.weight"] = inp[f"{old_prefix}.self_attn.layer_norm.weight"]
        attn_project_size = inp[f"{old_prefix}.self_attn.self_attn.project.weight"].size(0) // 3
        out[f"{prefix}.self_attention.project_q"] = inp[f"{old_prefix}.self_attn.self_attn.project.weight"][:attn_project_size]
        out[f"{prefix}.self_attention.project_k"] = inp[f"{old_prefix}.self_attn.self_attn.project.weight"][attn_project_size:2*attn_project_size]
        out[f"{prefix}.self_attention.project_v"] = inp[f"{old_prefix}.self_attn.self_attn.project.weight"][2*attn_project_size:]
        out[f"{prefix}.self_attention.attention_out"] = inp[f"{old_prefix}.self_attn.self_attn.dense.weight"]
        out[f"{prefix}.layernorm_before_ff.weight"] = inp[f"{old_prefix}.ff.layer_norm.weight"]
        out[f"{prefix}.ff.w_0"] = inp[f"{old_prefix}.ff.dense_relu_dense.wi_0.weight"]
        out[f"{prefix}.ff.w_1"] = inp[f"{old_prefix}.ff.dense_relu_dense.wi_1.weight"]
        out[f"{prefix}.ff.w_out"] = inp[f"{old_prefix}.ff.dense_relu_dense.wo.weight"]

    for i in tqdm(range(24), desc="Decoder"):
        prefix = f"dec_layers.{i}"
        old_prefix = f"decoder.blocks.{i}"
        out[f"{prefix}.layernorm_before_self_attention.weight"] = inp[f"{old_prefix}.self_attn.layer_norm.weight"]
        attn_project_size = inp[f"{old_prefix}.self_attn.self_attn.project.weight"].size(0) // 3
        out[f"{prefix}.self_attention.project_q"] = inp[f"{old_prefix}.self_attn.self_attn.project.weight"][:attn_project_size]
        out[f"{prefix}.self_attention.project_k"] = inp[f"{old_prefix}.self_attn.self_attn.project.weight"][attn_project_size:2*attn_project_size]
        out[f"{prefix}.self_attention.project_v"] = inp[f"{old_prefix}.self_attn.self_attn.project.weight"][2*attn_project_size:]
        out[f"{prefix}.self_attention.attention_out"] = inp[f"{old_prefix}.self_attn.self_attn.dense.weight"]
        
        out[f"{prefix}.layernorm_before_cross_attention.weight"] = inp[f"{old_prefix}.cross_attn.layer_norm.weight"]
        out[f"{prefix}.cross_attention.project_q"] = inp[f"{old_prefix}.cross_attn.cross_attn.project_q.weight"]
        attn_project_size = inp[f"{old_prefix}.cross_attn.cross_attn.project_kv.weight"].size(0) // 2
        out[f"{prefix}.cross_attention.project_k"] = inp[f"{old_prefix}.cross_attn.cross_attn.project_kv.weight"][:attn_project_size]
        out[f"{prefix}.cross_attention.project_v"] = inp[f"{old_prefix}.cross_attn.cross_attn.project_kv.weight"][attn_project_size:]
        out[f"{prefix}.cross_attention.attention_out"] = inp[f"{old_prefix}.cross_attn.cross_attn.dense.weight"]

        out[f"{prefix}.layernorm_before_ff.weight"] = inp[f"{old_prefix}.ff.layer_norm.weight"]
        out[f"{prefix}.ff.w_0"] = inp[f"{old_prefix}.ff.dense_relu_dense.wi_0.weight"]
        out[f"{prefix}.ff.w_1"] = inp[f"{old_prefix}.ff.dense_relu_dense.wi_1.weight"]
        out[f"{prefix}.ff.w_out"] = inp[f"{old_prefix}.ff.dense_relu_dense.wo.weight"]

    torch.save(out, "cpm2_ckpt.pt")

if __name__ == "__main__":
    main()