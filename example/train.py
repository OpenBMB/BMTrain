import torch
import bmtrain as bmt
from models import GPT
import time
from bmtrain import optim
from bmtrain.global_var import config
from bmtrain import inspect
from inspect_tools import lookup_output, custom_redirection
def main():
    bmt.init_distributed(
        seed=0,
        tp_size=1,
    )

    model = GPT(
        num_layers=8,
        vocab_size=10240, 
        dim_model=2560,
        dim_head=80,
        num_heads=32,
        dim_ff=8192,
        max_distance=1024,
        bias=True,
        dtype=torch.half
    )
    inspect_iter = -1
    bmt.load(model, "./ckpt-0.pt")
    bmt.print_rank("Model memory")
    bmt.print_rank(torch.cuda.memory_summary())
    bmt.synchronize()
    # data
    # generate dummy data for each rank
    torch.manual_seed(1234)
    batch_size = 2
    seq_len = 512
    global_batch = 2 * 16

    # for i in range(bmt.world_size()):
    #     sent = torch.randint(0, 10240, (batch_size, seq_len + 1))
    #     enc_length = torch.randint(128, seq_len, (batch_size,)).long().cuda()
    #     enc_input = sent[:, :-1].long().cuda()
    #     targets = sent[:, 1:].long().cuda()
    #     mask = torch.arange(seq_len).long().cuda()[None, :] < enc_length[:, None]
    #     targets = torch.where(
    #         mask,
    #         targets,
    #         torch.full_like(targets, -100, dtype=torch.long)
    #     )

    #     if i == bmt.rank():
    #         break
    if config['tp_size'] > 1:
        loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100, parallel=True)
    else:
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

    optimizer = optim.AdamOffloadOptimizer(model.parameters(), weight_decay=1e-2)
    lr_scheduler = bmt.lr_scheduler.Noam(optimizer, start_lr=1e-3, warmup_iter=40, end_iter=1000, num_iter=0)

    optim_manager = optim.OptimManager(loss_scale=2**20)
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    bmt.synchronize()
    
    avg_time_recorder = bmt.utils.AverageRecorder()
    avg_loss_recorder = bmt.utils.AverageRecorder()
    for iteration in range(10):
        # load data
        st = time.time()
        if iteration == inspect_iter:
            lookup_output(model)
        sum_loss = 0
        for micro in range(global_batch // batch_size):
            # for i in range(bmt.world_size()):
            sent = torch.randint(0, 10240, (batch_size, seq_len + 1))
            enc_length = torch.randint(128, seq_len, (batch_size,)).long().cuda()
            enc_input = sent[:, :-1].long().cuda()
            targets = sent[:, 1:].long().cuda()
            mask = torch.arange(seq_len).long().cuda()[None, :] < enc_length[:, None]
            targets = torch.where(
                mask,
                targets,
                torch.full_like(targets, -100, dtype=torch.long)
            )

            # if i == bmt.rank():
            #     break
 
        # with inspect.inspect_tensor() as inspector:
            pos = torch.arange(enc_input.size(1)).long().cuda().repeat(enc_input.size(0), 1)
            # if iteration == 4:
                # lookup_output(model)
            if iteration == inspect_iter:
                with custom_redirection("dp_ref.output"):
                    logits = model(
                        enc_input,
                        pos,
                        pos < enc_length[:, None]
                    )
            else:
                logits = model(
                    enc_input,
                    pos,
                    pos < enc_length[:, None]
                )
            batch, seq_len, vocab_out_size = logits.size()

            if config['tp_size'] > 1:
                loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets)
            else:
                loss = loss_func(logits.float().view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))
            global_loss = loss.item()
            optim_manager.backward(loss)
            sum_loss += global_loss 
        # print inspected tensors in the forward & backward pass
        # print parameters of the model
        if iteration % 100 == 0:
            bmt.print_rank(
                inspect.format_summary(
                    inspect.inspect_model(model, "*")
                )
            )
        optim_manager.step()
        optim_manager.zero_grad()
        # record time and loss
        iteration_time = time.time() - st

        avg_time_recorder.record(iteration_time)
        num_micro = global_batch // batch_size
        avg_loss_recorder.record(sum_loss/num_micro)
        # print time and loss
        bmt.print_rank(
            "| Iter: {:6d} | loss: {:.10f} average_loss: {:.4f} | lr: {:.4e} scale: {:10.4f} | time: {:.4f}".format(
                iteration,
                sum_loss / num_micro,
                avg_loss_recorder.value,
                lr_scheduler.current_lr,
                optim_manager.loss_scale,
                avg_time_recorder.value
            )
        )

        # save model
        if iteration % 1000 == 0:
            bmt.save(model, "ckpt-%d.pt" % iteration)
    
    bmt.save(model, "checkpoint.pt")

if __name__ == '__main__':
    main()
