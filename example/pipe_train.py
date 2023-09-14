import torch
import bmtrain as bmt
from models import GPTPipe
import time
from bmtrain import optim
from bmtrain.global_var import config
from bmtrain import inspect
from bmtrain.pipe import pipeline_forward_backward

def main():
    bmt.init_distributed(
        seed=0,
        tp_size=1,
        pipe_size=4,
    )

    model = GPTPipe(
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

    bmt.init_parameters(model)

    bmt.print_rank("Model memory")
    bmt.print_rank(torch.cuda.memory_summary())
    bmt.synchronize()

    # data
    # generate dummy data for each rank
    torch.manual_seed(1234)

    batch_size = 2 * 4
    seq_len = 512
    def data_loader(): 
        for i in range(1000):
            micro = 2
            sent = torch.randint(0, 10240, (micro, seq_len + 1))
            enc_length = torch.randint(128, seq_len, (micro,)).long().cuda()
            enc_input = sent[:, :-1].long().cuda()
            targets = sent[:, 1:].long().cuda()
            mask = torch.arange(seq_len).long().cuda()[None, :] < enc_length[:, None]
            targets = torch.where(
                mask,
                targets,
                torch.full_like(targets, -100, dtype=torch.long)
            )
            pos = torch.arange(enc_input.size(1)).long().cuda().repeat(enc_input.size(0), 1)
            yield enc_input, pos, pos<enc_length[:, None], targets
    
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
    bmt.init_parameters(model)
    for iteration in range(1):
        # load data
        st = time.time()
        global_loss = pipeline_forward_backward(model, data_loader(), batch_size) 
        
        # print inspected tensors in the forward & backward pass
        # print parameters of the model
        # if iteration % 100 == 0:
        #     bmt.print_rank(
        #         inspect.format_summary(
        #             inspector.get_summary()
        #         )
        #     )
        #     bmt.print_rank(
        #         inspect.format_summary(
        #             inspect.inspect_model(model, "*")
        #         )
            # )

        optim_manager.step()

        # record time and loss
        iteration_time = time.time() - st
        # avg_time_recorder.record(iteration_time)
        # avg_loss_recorder.record(global_loss)
        if global_loss is not None:
            print(global_loss)
        # print("hello")
        # print time and loss
        # if config['topology'].pipe_rank == config['topology'].pipe_size - 1:
        #     bmt.print_rank_pp(
        #         "| Iter: {:6d} | loss: {:.4f} average_loss: {:.4f} | lr: {:.4e} scale: {:10.4f} | time: {:.4f}".format(
        #             iteration,
        #             global_loss,
        #             avg_loss_recorder.value,
        #             lr_scheduler.current_lr,
        #             optim_manager.loss_scale,
        #             avg_time_recorder.value
        #         ), pipe_rank=config['pipe_size'] - 1
        #     )

    

if __name__ == '__main__':
    main()
