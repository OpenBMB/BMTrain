import torch
import bmtrain as bmt
from bmtrain.global_var import config
from bmtrain.distributed import all_reduce, all_gather

class ParallelCrossEntropyFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, target, label_smoothing=0.0):
        comm = config['tp_comm']
        rank = config['topology'].tp_id
        world_size = config['tp_size']

        # local max
        max_logits = torch.max(logits, dim=-1)[0]
        # global max
        max_logits = all_reduce(max_logits, op="max", comm=comm)

        logits = logits - max_logits.unsqueeze(dim=-1)

        partition_vocab_size = logits.size()[-1]
        vocab_start_index =  rank * partition_vocab_size
        vocab_end_index =  (rank + 1) * partition_vocab_size

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        logits_2d = logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                                 device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        # All reduce is needed to get the chunks from other GPUs.
        predicted_logits = all_reduce(predicted_logits, op="sum", comm=comm)
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = logits
        torch.exp(logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        sum_exp_logits = all_reduce(sum_exp_logits, op="sum", comm=comm)

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits.view(predicted_logits.shape)) - predicted_logits

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            """
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                                 device=grad_2d.device)

        softmax_update = 1.0 - target_mask.view(-1).float()

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / vocab_size
            grad_2d[arange_1d, :] -= smoothing * average_grad
        else:
            grad_2d[arange_1d, masked_target_1d] -= softmax_update

        grad_input.mul_(grad_output.flatten(0,1).unsqueeze(dim=-1))

        return grad_input, None, None


def parallel_cross_entropy_func(logits, target, label_smoothing=0.0):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks

    Arguments:
        logits: logits split across tensor parallel ranks dimension is [batch * seq_len, hidden_size].
        target: correct vocab ids of dimseion [batch * seq_len].
        lobal_smoothing: smoothing factor, must be in range [0.0, 1.0). default is 0.0.
    """
    out = ParallelCrossEntropyFunc.apply(logits.to(torch.float32), target, label_smoothing)
    return out

