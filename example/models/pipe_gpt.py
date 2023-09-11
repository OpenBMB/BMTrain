import torch
import bmtrain as bmt
from layers import TransformerEncoder, Layernorm, Embedding, TransformerEncoder
from bmtrain.global_var import config
class InputWrapper(bmt.DistributedModule):
    def __init__(self, module_list):
        super().__init__()

        self._module = {}
        for i in range(len(module_list)):
            self._module[str(i)] = module_list[i]
        
    def forward(self, *args):
        output_list = []
        for idx,i in enumerate(args):
            output_list.append(self._module[str(idx)](i))
        return sum(output_list)




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
            word_emb = bmt.nn.ParallelEmbedding(vocab_size, dim_model, dtype=dtype)
        else:
            word_emb = bmt.nn.PipeEmbedding(vocab_size, dim_model, dtype=dtype)
        pos_emb = Embedding(max_distance, dim_model, dtype=dtype)
        # self.inp_emb = InputWrapper([word_emb, pos_emb])
        blocklist = [word_emb]
        blocklist += [
                TransformerEncoder(
                    dim_model, dim_head, num_heads, dim_ff, bias, dtype
                )
             for _ in range(num_layers)]
        layernorm = Layernorm(dim_model, dtype=dtype)
        self.transformers = bmt.PipeDreamBlockList(
            blocklist,
        )
        if config['topology'].pipe_rank == config['topology'].pipe_size - 1:
            self.word_emb = word_emb

        if config['tp_size'] > 1:
            self.loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100, parallel=True)
        else:
            self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
    def forward(self,
            input : torch.LongTensor,   # (batch, seq_len)
            pos : torch.LongTensor,     # (batch, seq_len)
            mask : torch.BoolTensor,    # (batch, seq_len)
            target: torch.LongTensor,
        ) -> torch.Tensor:
        mask_2d = mask[:, None, :] & mask[:, :, None]   # (batch, seq_len, seq_len)
        mask_2d = mask_2d & (pos[:, None, :] >= pos[:, :, None])


        # for layer in self.transformers:
        out = self.transformers(input, mask_2d, None)
        if config['topology'].pipe_rank == config['topology'].pipe_size - 1:
            if config['tp_size'] > 1:
                logits = self.word_emb.projection(out)
            else:
                logits = self.word_emb(out, projection=True)

            return self.loss_func(logits, target)
        else:
            return out, pos, mask, target

    def preprocess_func(self, inp):
        if config['topology'].pipe_rank == 0:
            inp_pos = inp[:1]
            return self.transformers['0'](*inp_pos), *inp[1:]
        else:
            return None
        
        