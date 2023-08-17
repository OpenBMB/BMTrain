from typing import Optional
import torch
import bmtrain as bmt
from layers import Linear
import math
from .flash_triton import FlashAttnFunc
class Attention(bmt.DistributedModule):
    def __init__(
        self,
        dim_model: int,
        dim_head: int,
        num_heads: int,
        bias: bool = False,
        dtype: torch.dtype = torch.half,
        dropout_p: Optional[float] = None,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.dim_head = dim_head

        self.project_q = Linear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype, )
        self.project_k = Linear(self.dim_model, self.num_kv_heads * self.dim_head, dtype=dtype, )
        self.project_v = Linear(self.dim_model, self.num_kv_heads * self.dim_head, dtype=dtype, )

        self.attention_out = Linear(self.num_heads * self.dim_head, self.dim_model, dtype=dtype, )


        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=dropout_p)
            self.dropout_p = dropout_p
        else:
            self.dropout = None

        # if use_flash_attn:
        #     self.core_attention_flash = FlashSelfAttention(causal=False, attention_dropout=0.0)
        self.use_flash_attn = use_flash_attn

    def forward(
        self,
        hidden_q: torch.Tensor,
        hidden_kv: torch.Tensor,
        attention_mask: torch.BoolTensor,
    ):
        """This model inherits from bmt.DistributedModule.
        Args:
            hidden_q (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            hidden_kv (:obj:`torch.Tensor` of shape ``(batch, len_k, dim_model)``): Length of input sequence before padding.
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, len_q, len_k)``): Used to avoid performing attention on padding token indices.
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, len_q, len_k)`` or ``(1, num_heads, len_k, len_q)``): Provide positional information about tensor `key_value` and `query`.
        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): The attention output.
        """  # noqa: E501

        batch_size = hidden_q.size(0)
        len_q = hidden_q.size(1)
        len_k = hidden_kv.size(1)

        h_q = self.project_q(hidden_q)
        h_k = self.project_k(hidden_kv)
        h_v = self.project_v(hidden_kv)

           # 
            # if self.head_groups != 1:
            #     h_k = h_k[:, :, :, None, :].expand(batch_size, len_k, self.num_kv_heads, self.head_groups, self.dim_head).reshape(batch_size, len_k, self.num_heads, self.dim_head)
            #     h_v = h_v[:, :, :, None, :].expand(batch_size, len_k, self.num_kv_heads, self.head_groups, self.dim_head).reshape(batch_size, len_k, self.num_heads, self.dim_head)


            # h_q = h_q.permute(0, 2, 1, 3).contiguous()
            # h_k = h_k.permute(0, 2, 1, 3).contiguous()
            # h_v = h_v.permute(0, 2, 1, 3).contiguous()

            # B, S, H, D
            # score = self.core_attention_flash(
                # h_q, h_k, h_v, attention_mask=attention_mask, length_mask=length_mask, context_mask=context_mask
            # )
        if attention_mask is not None:
            h_q = h_q.view(batch_size, len_q, self.num_heads, self.dim_head)  # .permute(0, 2, 1, 3)
            h_k = h_k.view(batch_size, len_k, self.num_kv_heads, self.dim_head)  # .permute(0, 2, 1, 3)
            h_v = h_v.view(batch_size, len_k, self.num_kv_heads, self.dim_head)  # .permute(0, 2, 1, 3)
            mask = attention_mask
            mask_bias = torch.zeros_like(attention_mask, device="cuda", dtype=torch.float16)  # 创建与mask形状相同的全零张量
            mask_bias[mask == False] -= torch.inf
            mask_bias = mask_bias.unsqueeze(1)
            # if hasattr(self, "_offload_hook"):
            #     pack, unpack = self._offload_hook
            #     torch._C._autograd._push_saved_tensors_default_hooks(
            #         pack, unpack
            #     )
            score = FlashAttnFunc.apply(h_q, h_k, h_v, mask_bias, False, None)
            # if hasattr(self, "_offload_hook"):
            #     torch._C._autograd._pop_saved_tensors_default_hooks()
            score = score.view(batch_size, len_q, self.num_heads * self.dim_head)
            

        score = self.attention_out(score)

        return score

#class Attention(bmt.DistributedModule):
#    def __init__(self, 
#            dim_model : int, dim_head : int,
#            num_heads : int, bias : bool = True,
#            dtype = None
#        ) -> None:
#        super().__init__()
#
#        self.project_q = Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
#        self.project_k = Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
#        self.project_v = Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
#
#        self.project_out = Linear(dim_head * num_heads, dim_model, bias=bias, dtype=dtype)
#
#        self.softmax = torch.nn.Softmax(dim=-1)
#        self.num_heads = num_heads
#        self.dim_head = dim_head
#        self.dim_model = dim_model
#    
#    def forward(self, 
#            hidden_q : torch.Tensor,        # (batch_size, seq_q, dim_model)
#            hidden_kv : torch.Tensor,       # (batch_size, seq_kv, dim_model)
#            mask : torch.BoolTensor,        # (batch_size, seq_q, seq_kv)
#            position_bias : Optional[torch.Tensor] = None,   # (batch, num_heads, seq_q, seq_kv)
#        ) -> torch.Tensor:
#        batch_size, seq_q, dim_model = hidden_q.size()
#        seq_kv = hidden_kv.size(1)
#
#        h_q : torch.Tensor = self.project_q(hidden_q)
#        h_k : torch.Tensor = self.project_k(hidden_kv)
#        h_v : torch.Tensor = self.project_v(hidden_kv)
#
#        h_q = h_q.view(batch_size, seq_q, self.num_heads, self.dim_head)
#        h_k = h_k.view(batch_size, seq_kv, self.num_heads, self.dim_head)
#        h_v = h_v.view(batch_size, seq_kv, self.num_heads, self.dim_head)
#
#        h_q = h_q.permute(0, 2, 1, 3).contiguous()
#        h_k = h_k.permute(0, 2, 1, 3).contiguous()
#        h_v = h_v.permute(0, 2, 1, 3).contiguous()
#
#        h_q = h_q.view(batch_size * self.num_heads, seq_q, self.dim_head)
#        h_k = h_k.view(batch_size * self.num_heads, seq_kv, self.dim_head)
#        h_v = h_v.view(batch_size * self.num_heads, seq_kv, self.dim_head)
#
#        score = torch.bmm(
#            h_q, h_k.transpose(1, 2)
#        )
#        score = score / math.sqrt(self.dim_head)
#
#        score = score.view(batch_size, self.num_heads, seq_q, seq_kv)
#
#        if position_bias is not None:
#            score = score + position_bias.view(batch_size, self.num_heads, seq_q, seq_kv)
#        
#        score = torch.where(
#            mask.view(batch_size, 1, seq_q, seq_kv),
#            score,
#            torch.scalar_tensor(float('-inf'), device=score.device, dtype=score.dtype)
#        )
#
#        score = torch.where(
#            mask.view(batch_size, 1, seq_q, seq_kv),
#            self.softmax(score),
#            torch.scalar_tensor(0, device=score.device, dtype=score.dtype)
#        )
#
#        score = score.view(batch_size * self.num_heads, seq_q, seq_kv)
#
#        h_out = torch.bmm(
#            score, h_v
#        )
#        h_out = h_out.view(batch_size, self.num_heads, seq_q, self.dim_head)
#        h_out = h_out.permute(0, 2, 1, 3).contiguous()
#        h_out = h_out.view(batch_size, seq_q, self.num_heads * self.dim_head)
#
#        attn_out = self.project_out(h_out)
#        return attn_out
        

        


