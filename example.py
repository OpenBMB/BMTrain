import torch
import BMPretrain as bmp

class Config:
    DTYPE = torch.float16
    
    NUM_HEAD = 64
    DIM_HEAD = 64
    DIM_HIDDEN = 4096
    DIM_FF = 10240

    DROPOUT_RATE = 0.0
    

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_q : torch.Tensor, weight_q : torch.Tensor, hidden_k : torch.Tensor, weight_k : torch.Tensor, num_head : int):
        # hidden_q: (batch_size, seq_q, hidden_size)
        # weight_q: (num_head * dim_head, hidden_size)
        # hidden_k: (batch_size, seq_k, hidden_size)
        # weight_k: (num_head * dim_head, hidden_size)
        # output: (batch_size, num_head, seq_q, seq_k)

        # (1, num_head * dim_head, hidden_size) @ (batch_size, seq_q, hidden_size)T
        batch_size = hidden_q.size(0) 
        dim_head = weight_q.size(0) // num_head

        query = bmp.layers.bmm(weight_q.unsqueeze(0), False, hidden_q, True, False) # (batch_size, num_head * dim_head, seq_q)
        query = query.view(batch_size * num_head, dim_head, query.size(-1)) # (batch_size, num_head, dim_head, seq_q)

        key = bmp.layers.bmm(weight_k.unsqueeze(0), False, hidden_k, True, False) # (batch_size, num_head * dim_head, seq_k)
        key = key.view(batch_size * num_head, dim_head, key.size(-1)) # (batch_size, num_head, dim_head, seq_k)

        # (batch_size * num_head, dim_head, seq_q)T @ (batch_size * num_head, dim_head, seq_k)
        out = bmp.layers.bmm(query, True, key, False, False) # (batch_size * num_head, seq_q, seq_k)
        return out.view(batch_size, num_head, out.size(-2), out.size(-1))

    @staticmethod
    def backward(ctx, grad_output):
        pass    