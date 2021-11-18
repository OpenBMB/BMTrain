from typing import Optional
import torch
import bmpretrain as bmp
import cpm_kernels.torch as ct

class Attention(bmp.DistributedModule):
    def __init__(self, dim_model : int, num_heads : int, dim_head : int, int8=True, dtype=torch.half):
        super().__init__()

        self.project_q = bmp.DistributedParameter(torch.empty(num_heads * dim_head, dim_model, dtype=dtype), init_method=bmp.ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=0.02))
        self.project_k = bmp.DistributedParameter(torch.empty(num_heads * dim_head, dim_model, dtype=dtype), init_method=bmp.ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=0.02))
        self.project_v = bmp.DistributedParameter(torch.empty(num_heads * dim_head, dim_model, dtype=dtype), init_method=bmp.ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=0.02))

        self.attention_out = bmp.DistributedParameter(torch.empty(dim_model, num_heads * dim_head, dtype=dtype), init_method=bmp.ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=0.02))
    
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
        """
        Args:
            hidden_q : (batch, dim_model, len_q)    fp16
            hidden_kv : (batch, dim_model, len_k)   fp16
            mask : (batch, len_k, len_q)            fp16
            position_bias : (num_heads, len_k, len_q)   fp16
        Returns:
            out : (batch, dim_model, len_q)         fp16
        """
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
