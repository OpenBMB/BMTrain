from typing import Optional
import torch
import cpm_kernels.torch as ct

from .attention import Attention
from .layernorm import LayerNorm
from .feedforward import FeedForward
import bmpretrain as bmp

class TransformerEncoder(torch.nn.Module):
    def __init__(self, dim_model : int, num_heads : int, dim_head : int, dim_ff : int, eps : float, int8=True, dtype=torch.half):
        super().__init__()

        self.layernorm_before_attention = LayerNorm(dim_model, eps, bias=False, dtype=dtype)
        self.self_attention = Attention(dim_model, num_heads, dim_head, int8=int8, dtype=dtype)

        self.layernorm_before_ff = LayerNorm(dim_model, eps, bias=False, dtype=dtype)
        self.ff = FeedForward(dim_model, dim_ff, int8=int8, dtype=dtype)

    def forward(self,
            hidden_state : torch.Tensor,    # (batch, hidden_size, seq_len)
            mask : torch.Tensor,            # (batch, seq_len, seq_len)
            position_bias : torch.Tensor,   # (num_heads, seq_len, seq_len)
        ):
        """
        Args:
            hidden_state: (batch, hidden_size, seq_len)     fp16
            mask: (batch, seq_len, seq_len)                 fp16
            position_bias: (num_heads, seq_len, seq_len)    fp16
        Returns:
            out: (batch, hidden_size, seq_len)              fp16
        """
        bmp.inspect.record_tensor(hidden_state, "hidden_state", group="encoder")
        x = self.layernorm_before_attention(hidden_state)
        x = self.self_attention(x, x, mask, position_bias)
        hidden_state = ct.element_add(hidden_state, x)      # hidden_state = hidden_state + x
        

        x = self.layernorm_before_ff(hidden_state)
        x = self.ff(x)
        hidden_state = ct.element_add(hidden_state, x)      # hidden_state = hidden_state + x

        return hidden_state

class TransformerDecoder(torch.nn.Module):
    def __init__(self, dim_model : int, num_heads : int, dim_head : int, dim_ff : int, eps : float, int8=True, dtype=torch.half):
        super().__init__()

        self.layernorm_before_self_attention = LayerNorm(dim_model, eps, bias=False, dtype=dtype)
        self.self_attention = Attention(dim_model, num_heads, dim_head, int8=int8, dtype=dtype)

        self.layernorm_before_cross_attention = LayerNorm(dim_model, eps, bias=False, dtype=dtype)
        self.cross_attention = Attention(dim_model, num_heads, dim_head, int8=int8, dtype=dtype)

        self.layernorm_before_ff = LayerNorm(dim_model, eps, bias=False, dtype=dtype)
        self.ff = FeedForward(dim_model, dim_ff, int8=int8, dtype=dtype)

    def forward(self,
            hidden_state : torch.Tensor,                # (batch, hidden_size, seq_q)
            encoder_output : torch.Tensor,              # (batch, hidden_size, seq_k)
            mask_self_attn : torch.Tensor,              # (batch, seq_q, seq_q)
            mask_corss_attn : torch.Tensor,             # (batch, seq_k, seq_q)
            self_attn_bias : Optional[torch.Tensor],    # (num_heads, seq_q, seq_q)
            cross_attn_bias : Optional[torch.Tensor],   # (num_heads, seq_k, seq_q)
        ):
        """
        Args:
            hidden_state: (batch, hidden_size, seq_q)       fp16
            encoder_output: (batch, hidden_size, seq_k)     fp16
            mask_self_attn: (batch, seq_q, seq_q)           fp16
            mask_corss_attn: (batch, seq_k, seq_q)          fp16
            self_attn_bias: (num_heads, seq_q, seq_q)       fp16
            cross_attn_bias: (num_heads, seq_k, seq_q)      fp16
        Returns:
            out: (batch, hidden_size, seq_q)                fp16
        """
        bmp.inspect.record_tensor(hidden_state, "hidden_state", group="decoder")
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
