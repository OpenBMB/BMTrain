import torch
import bmtrain as bmt
from layers import TransformerEncoder, Layernorm, Embedding, TransformerEncoder

class GPT(bmt.DistributedModule):
    def __init__(self,
            num_layers : int, vocab_size : int,
            dim_model : int, dim_head : int, num_heads : int, dim_ff : int,
            max_distance : int,
            bias : bool = True, dtype = None
        ) -> None:
        super().__init__()

        self.max_distance = max_distance

        self.word_emb = Embedding(vocab_size, dim_model, dtype=dtype)
        self.pos_emb = Embedding(max_distance, dim_model, dtype=dtype)
        
        self.transformers = bmt.TransformerBlockList([
            bmt.CheckpointBlock(
                TransformerEncoder(
                    dim_model, dim_head, num_heads, dim_ff, bias, dtype
                )
            )
            for _ in range(num_layers)
        ])
        # self.transformers = [bmt.CheckpointBlock(
        #         TransformerEncoder(
        #             dim_model, dim_head, num_heads, dim_ff, bias, dtype
        #         )
        # )
        #     for _ in range(num_layers)
        # ]

        self.layernorm = Layernorm(dim_model, dtype=dtype)

    def forward(self,
            input : torch.LongTensor,   # (batch, seq_len)
            pos : torch.LongTensor,     # (batch, seq_len)
            mask : torch.BoolTensor,    # (batch, seq_len)
        ) -> torch.Tensor:

        mask_2d = mask[:, None, :] & mask[:, :, None]   # (batch, seq_len, seq_len)
        mask_2d = mask_2d & (pos[:, None, :] >= pos[:, :, None])

        input_emb = self.pos_emb(pos) + self.word_emb(input)
        # for i in range(len(self.transformers)):
        #     input_emb = self.transformers[i](input_emb, mask_2d, None)
        input_emb = self.transformers(input_emb, mask_2d, None)
        out = self.layernorm(input_emb)

        logits = self.word_emb(out, projection=True)
        bmt.inspect.record_tensor(logits, "logits")

        return logits