import torch
import bmpretrain as bmp
from models import GPT
import time

def main():
    bmp.init_distributed(
        seed=0,
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

    bmp.init_parameters(model)
    # print_inspect(model, "*")

    bmp.print_rank("Model memory")
    bmp.print_rank(torch.cuda.memory_summary())
    bmp.synchronize()

    # data
    # generate dummy data for each rank

    batch_size = 2
    seq_len = 512

    for i in range(bmp.world_size()):
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

        if i == bmp.rank():
            break
    
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = bmp.optim.AdamOffloadOptimizer(model.parameters(), weight_decay=1e-2, scale=2**20)
    lr_scheduler = bmp.lr_scheduler.Noam(optimizer, start_lr=1e-3, warmup_iter=40, end_iter=1000, num_iter=0)

    bmp.synchronize()
    
    avg_time_recorder = bmp.utils.AverageRecorder()
    avg_loss_recorder = bmp.utils.AverageRecorder()

    for iteration in range(1000):
        # load data
        st = time.time()
        optimizer.zero_grad()

        with bmp.inspect.inspect_tensor() as inspector:
            pos = torch.arange(enc_input.size(1)).long().cuda().repeat(enc_input.size(0), 1)
            logits = model(
                enc_input,
                pos,
                pos < enc_length[:, None]
            )
            batch, seq_len, vocab_out_size = logits.size()

            loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))
        
            global_loss = bmp.sum_loss(loss).item()

            loss = optimizer.loss_scale(loss)
            loss.backward()
        
        # print inspected tensors in the forward & backward pass
        # print parameters of the model
        if iteration % 100 == 0:
            bmp.print_rank(
                bmp.inspect.format_summary(
                    inspector.get_summary()
                )
            )
            bmp.print_rank(
                bmp.inspect.format_summary(
                    bmp.inspect.inspect_model(model, "*")
                )
            )
        

        bmp.optim_step(optimizer, lr_scheduler)

        # record time and loss
        iteration_time = time.time() - st

        avg_time_recorder.record(iteration_time)
        avg_loss_recorder.record(global_loss)

        # print time and loss
        bmp.print_rank(
            "| Iter: {:6d} | loss: {:.4f} average_loss: {:.4f} | lr: {:.4e} scale: {:10.4f} | time: {:.4f}".format(
                iteration,
                global_loss,
                avg_loss_recorder.value,
                lr_scheduler.current_lr,
                optimizer.scale,
                avg_time_recorder.value
            )
        )

        # save model
        if iteration % 1000 == 0:
            bmp.save(model, "ckpt-%d.pt" % iteration)
    
    bmp.save(model, "checkpoint.pt")

if __name__ == '__main__':
    main()