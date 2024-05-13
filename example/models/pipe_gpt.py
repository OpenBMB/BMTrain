import torch
import bmtrain as bmt
from layers import TransformerEncoder, Layernorm, Embedding, TransformerEncoder
from bmtrain.global_var import config

class GPTPipe(bmt.DistributedModule):
    def __init__(self,
            num_layers : int, vocab_size : int,
            dim_model : int, dim_head : int, num_heads : int, dim_ff : int,
            max_distance : int,
            bias : bool = True, dtype = None
        ) -> None:
        super().__init__()

        self.max_distance = max_distance

        if config['tp_size'] > 1:
            word_emb = bmt.nn.VPEmbedding(vocab_size, dim_model, dtype=dtype)
        else:
            word_emb = Embedding(vocab_size, dim_model, dtype=dtype)
        pos_emb = Embedding(max_distance, dim_model, dtype=dtype)
        blocklist = []
        blocklist += [
                TransformerEncoder(
                    dim_model, dim_head, num_heads, dim_ff, bias, dtype
                )
             for _ in range(num_layers)]
        layernorm = Layernorm(dim_model, dtype=dtype)
        self.transformers = bmt.PipeDreamBlockList(
            blocklist,
        )
        self.pos_emb = self.transformers.add_head(pos_emb)
        self.layernorm = self.transformers.add_tail(layernorm)
        self.word_emb = self.transformers.add_head_tail(word_emb)

    def get_blocklist(self):
        return self.transformers

    def forward(self,
            input : torch.LongTensor,   # (batch, seq_len)
            pos : torch.LongTensor,     # (batch, seq_len)
            mask : torch.BoolTensor,    # (batch, seq_len)
        ) -> torch.Tensor:
        mask_2d = mask[:, None, :] & mask[:, :, None]   # (batch, seq_len, seq_len)
        mask_2d = mask_2d & (pos[:, None, :] >= pos[:, :, None])


        # for layer in self.transformers:
        out = self.transformers(input, mask_2d, None)
        if bmt.config['topology'].is_last_rank():
            out = self.layernorm(out)
            out = self.word_emb(out, True)
        return out

    def preprocess_func(self, inp):
        if config['topology'].pipe_rank == 0:
            inp_id = inp[0]
            pos = inp[1]
            out = self.pos_emb(pos) + self.word_emb(inp_id) 
            return out
        else:
            return None
        
        
