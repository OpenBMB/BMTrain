from typing import Optional
import torch
import cpm_kernels.torch as ct


LayerNorm = ct.LayerNorm
PositionEmbedding = ct.PositionEmbedding
Embedding = ct.Embedding

class FeedForward(torch.nn.Module):
    def __init__(self, dim_model : int, dim_ff : int, int8=True):
        super().__init__()
        self.w_0 = torch.nn.Parameter(torch.Tensor(dim_ff, dim_model))
        self.w_1 = torch.nn.Parameter(torch.Tensor(dim_ff, dim_model))
        self.w_out = torch.nn.Parameter(torch.Tensor(dim_model, dim_ff))

        self.int8 = int8
    
    def forward(self, x):
        """
        x : (batch, hidden_size, seq_len)
        """
        # (1#batch, dim_ff, dim_model) @ (batch, dim_model, seq_len) = (batch, dim_ff, seq_len)
        gelu_score = ct.gelu(ct.bmm(self.w_0.unsqueeze(0), False, x, False, int8=self.int8))
        hidden_out = ct.bmm(self.w_1.unsqueeze(0), False, x, False, int8=self.int8)
        
        # (batch, dim_ff, seq_len)
        x = ct.element_mul(gelu_score, hidden_out)

        # (1#batch, dim_model, dim_ff) @ (batch, dim_ff, seq_len) = (batch, dim_model, seq_len)
        x = ct.bmm(self.w_out.unsqueeze(0), False, x, False, int8=self.int8)
        return x

class Attention(torch.nn.Module):
    def __init__(self, dim_model : int, num_heads : int, dim_head : int, int8=True):
        super().__init__()

        self.project_q = torch.nn.Parameter(torch.Tensor(num_heads * dim_head, dim_model))
        self.project_k = torch.nn.Parameter(torch.Tensor(num_heads * dim_head, dim_model))
        self.project_v = torch.nn.Parameter(torch.Tensor(num_heads * dim_head, dim_model))

        self.attention_out = torch.nn.Parameter(torch.Tensor(dim_model, num_heads * dim_head))
    
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.int8 = int8
    
    def forward(self, 
            hidden_q : torch.Tensor,                # (batch, dim_model, len_q)
            hidden_kv : torch.Tensor,               # (batch, dim_model, len_k)
            mask : torch.Tensor,                    # (batch, len_k, len_q)
            position_bias : Optional[torch.Tensor]  # (num_heads, len_k, len_q)
        ):
        batch_size = hidden_q.size(0)
        len_q = hidden_q.size(2)
        len_k = hidden_kv.size(2)

        # (1#batch, num_heads * dim_head, dim_model) @ (batch, dim_model, len_q) = (batch, num_heads * dim_head, len_q)
        h_q = ct.bmm(self.project_q.unsqueeze(0), False, hidden_q, False, int8=self.int8)
        h_k = ct.bmm(self.project_k.unsqueeze(0), False, hidden_kv, False, int8=self.int8)
        h_v = ct.bmm(self.project_v.unsqueeze(0), False, hidden_kv, False, int8=self.int8)

        # view (batch * num_heads, dim_head, length)
        h_q = h_q.view(batch_size * self.num_heads, self.dim_head, -1)
        h_k = h_k.view(batch_size * self.num_heads, self.dim_head, -1)
        h_v = h_v.view(batch_size * self.num_heads, self.dim_head, -1)

        # (batch * num_heads, dim_head, len_k)T @ (batch * num_heads, dim_head, len_q) = (batch * num_heads, len_k, len_q)
        score = ct.bmm( h_k, True, h_q, False, int8=False)  # use FP 16 here
        
        # (batch, num_heads, len_k, len_q) 
        score = score.view(batch_size, self.num_heads, len_k, len_q)
        if position_bias is not None:
            score = ct.batched_add(
                score,   
                position_bias
            )
        
        # (batch, num_heads, len_k * len_q)
        masked_score = ct.mask(
            score.view(batch_size, self.num_heads, -1),
            mask.view(batch_size, -1),
            float("-inf")
        )

        # (batch * num_heads, len_k, len_q)
        masked_score = masked_score.view(batch_size * self.num_heads, len_k, len_q)

        # (batch * num_heads, len_k, len_q)
        masked_score = ct.softmax(masked_score) # softmax along len_k

        # (batch * num_heads, dim_head, len_k) @ (batch * num_heads, len_k, len_q) = (batch * num_heads, dim_head, len_q)
        attention_result = ct.bmm(h_v, False, masked_score, False, int8=False)  # use FP 16 here

        attention_result = attention_result.view(batch_size, self.num_heads * self.dim_head, len_q)

        # (1#batch, dim_model, num_heads * dim_head) @ (batch, num_heads * dim_head, len_q) = (batch, dim_model, len_q)
        attention_out = ct.bmm(self.attention_out.unsqueeze(0), False, attention_result, False, int8=self.int8)

        return attention_out

class TransformerEncoder(torch.nn.Module):
    def __init__(self, dim_model : int, num_heads : int, dim_head : int, dim_ff : int, eps : float, int8=True):
        super().__init__()

        self.layernorm_before_attention = LayerNorm(dim_model, eps, bias=False)
        self.self_attention = Attention(dim_model, num_heads, dim_head, int8=int8)

        self.layernorm_before_ff = LayerNorm(dim_model, eps, bias=False)
        self.ff = FeedForward(dim_model, dim_ff, int8=int8)

    def forward(self,
            hidden_state : torch.Tensor,    # (batch, hidden_size, seq_len)
            mask : torch.Tensor,            # (batch, seq_len, seq_len)
            position_bias : torch.Tensor,   # (num_heads, seq_len, seq_len)
        ):
        
        x = self.layernorm_before_attention(hidden_state)
        x = self.self_attention(x, x, mask, position_bias)
        hidden_state = ct.element_add(hidden_state, x)

        x = self.layernorm_before_ff(hidden_state)
        x = self.ff(x)
        hidden_state = ct.element_add(hidden_state, x)

        return hidden_state

class TransformerDecoder(torch.nn.Module):
    def __init__(self, dim_model : int, num_heads : int, dim_head : int, dim_ff : int, eps : float, int8=True):
        super().__init__()

        self.layernorm_before_self_attention = LayerNorm(dim_model, eps, bias=False)
        self.self_attention = Attention(dim_model, num_heads, dim_head, int8=int8)

        self.layernorm_before_cross_attention = LayerNorm(dim_model, eps, bias=False)
        self.cross_attention = Attention(dim_model, num_heads, dim_head, int8=int8)

        self.layernorm_before_ff = LayerNorm(dim_model, eps, bias=False)
        self.ff = FeedForward(dim_model, dim_ff, int8=int8)

    def forward(self,
            hidden_state : torch.Tensor,                # (batch, hidden_size, seq_dec)
            encoder_output : torch.Tensor,              # (batch, hidden_size, seq_enc)
            mask_self_attn : torch.Tensor,              # (batch, seq_dec, seq_dec)
            mask_corss_attn : torch.Tensor,             # (batch, seq_enc, seq_dec)
            self_attn_bias : Optional[torch.Tensor],    # (num_heads, seq_dec, seq_dec)
            cross_attn_bias : Optional[torch.Tensor],   # (num_heads, seq_enc, seq_dec)
        ):
        
        x = self.layernorm_before_self_attention(hidden_state)
        x = self.self_attention(x, x, mask_self_attn, self_attn_bias)
        hidden_state = ct.element_add(hidden_state, x)

        x = self.layernorm_before_cross_attention(hidden_state)
        x = self.cross_attention(x, encoder_output, mask_corss_attn, cross_attn_bias)
        hidden_state = ct.element_add(hidden_state, x)

        x = self.layernorm_before_ff(hidden_state)
        x = self.ff(x)
        hidden_state = ct.element_add(hidden_state, x)

        return hidden_state

class T5(torch.nn.Module):
    def __init__(self, 
            num_enc : int, num_dec : int,                                       # layers
            dim_model : int, num_heads : int, dim_head : int, dim_ff : int,     # shapes
            vocab_input_size : int, vocab_output_size : int,                    # inputs
            position_bias_num_buckets : int, position_bias_max_distance : int,
            eps : float = 1e-6, int8 : bool = True
        ):
        super().__init__()
        
        self.num_enc = num_enc
        self.num_dec = num_dec

        self.enc_layers = torch.nn.ModuleList([
            TransformerEncoder(dim_model, num_heads, dim_head, dim_ff, eps, int8=int8)
            for _ in range(num_enc)
        ])

        self.dec_layers = torch.nn.ModuleList([
            TransformerDecoder(dim_model, num_heads, dim_head, dim_ff, eps, int8=int8)
            for _ in range(num_dec)
        ])

        self.layernorm_after_enc = LayerNorm(dim_model, eps, bias=False)
        self.layernorm_after_dec = LayerNorm(dim_model, eps, bias=False)

        self.input_embedding = Embedding(vocab_input_size, dim_model)
        self.output_embedding = torch.nn.Parameter(torch.Tensor(vocab_output_size, dim_model))

        self.position_bias_enc = PositionEmbedding(num_heads, position_bias_num_buckets, position_bias_max_distance, bidirectional=True)
        self.position_bias_dec = PositionEmbedding(num_heads, position_bias_num_buckets, position_bias_max_distance, bidirectional=False)
    
    def forward(self, 
            enc_input : torch.Tensor,       # (batch, seq_enc),
            enc_length : torch.Tensor,      # (batch),

            dec_input : torch.Tensor,       # (batch, seq_dec),
            dec_length : torch.Tensor,      # (batch),
        ):
        batch = enc_input.size(0)
        seq_enc = enc_input.size(1)
        seq_dec = dec_input.size(1)

        device = enc_input.device

        enc_mask_1d = torch.arange(seq_enc, device=device)[None, :].repeat(batch, 1) < enc_length[:, None]
        dec_mask_1d = torch.arange(seq_dec, device=device)[None, :].repeat(batch, 1) < dec_length[:, None]
        directional_mask = torch.arange(seq_dec, device=device).view(-1, 1) <= torch.arange(seq_dec, device=device)

        # (batch, seq_enc, seq_enc)
        enc_mask = enc_mask_1d.view(batch, seq_enc, 1) & enc_mask_1d.view(batch, 1, seq_enc)

        # (batch, seq_dec, seq_dec)
        dec_mask = dec_mask_1d.view(batch, seq_dec, 1) & dec_mask_1d.view(batch, 1, seq_dec) & directional_mask.view(1, seq_dec, seq_dec)

        # (batch, seq_enc, seq_dex)
        cross_mask = enc_mask_1d.view(batch, seq_enc, 1) & dec_mask_1d.view(batch, 1, seq_dec)

        # (num_heads, seq_enc, seq_enc)
        position_bias_enc = self.position_bias_enc(seq_enc, seq_enc)

        # (num_heads, seq_dec, seq_dec)
        position_bias_dec = self.position_bias_dec(seq_dec, seq_dec)

        # (batch, dim_model, seq_enc)
        hidden_enc = self.input_embedding(enc_input)

        for i in range(self.num_enc):
            hidden_enc = self.enc_layers[i](
                hidden_enc,
                enc_mask,
                position_bias_enc,
            )
        hidden_enc = self.layernorm_after_enc(hidden_enc)

        hidden_dec = self.input_embedding(dec_input)
        for i in range(self.num_enc):
            hidden_dec = self.dec_layers[i](
                hidden_dec,
                hidden_enc,
                dec_mask,
                cross_mask,
                position_bias_dec,
                None,   # no cross attention mask
            )
        # (batch, dim_model, seq_dec)
        hidden_dec = self.layernorm_after_dec(hidden_dec)

        # (1#batch, vocab_output_size, dim_model) @ (batch, dim_model, seq_dec) = (batch, vocab_output_size, seq_dec)
        logits = ct.bmm(self.output_embedding.unsqueeze(0), False, hidden_dec, False, int8=False)

        # (batch, seq_dec, vocab_output_size)
        logits = ct.transpose(logits)   # .transpose(1, 2)

        return logits


def init_all(model : T5):
    state_dict = model.state_dict()

    for kw in state_dict.keys():
        torch.nn.init.normal_(state_dict[kw], 0, 0.1)
    model.load_state_dict(state_dict)
    return model

def main(): 
    with torch.cuda.device(3):
        mini_T5 = T5(
            num_enc=6, num_dec=6,
            dim_model=512, num_heads=8, dim_head=64, dim_ff=2048,
            vocab_input_size=32, vocab_output_size=32,
            position_bias_num_buckets=4, position_bias_max_distance=128,
            eps=1e-6, int8=True
        )
        init_all(mini_T5)
        mini_T5 = mini_T5.cuda().half()

        enc_input = torch.randint(0, 30, (4, 16))   # (4, 16)
        enc_length = torch.randint(1, 16, (4,))     # (4,)

        dec_input = torch.randint(0, 30, (4, 16))   # (4, 16)
        dec_length = torch.randint(1, 16, (4,))     # (4,)

        logits = mini_T5(
            enc_input.cuda().int(), 
            enc_length.cuda(),
            dec_input.cuda().int(), 
            dec_length.cuda(),
        )
        print(logits)

        logits.backward(
            gradient=torch.randn(logits.size(), device="cuda", dtype=torch.half)
        )
        print(mini_T5.input_embedding.weight.grad)

if __name__ == "__main__":
    main()

