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
        pipe_size=4,
        tp_size=1,
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
        dtype=torch.float16
    )
    bmt.init_parameters(model)
    bmt.print_rank("Model memory")
    bmt.print_rank(torch.cuda.memory_summary())
    bmt.synchronize()

    # data
    # generate dummy data for each rank
    torch.manual_seed(1234)
    micro = 2
    num_micros = 16
    batch_size = micro * num_micros
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
        loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    optimizer = optim.AdamOffloadOptimizer(model.parameters(), weight_decay=1e-2)
    lr_scheduler = bmt.lr_scheduler.Noam(optimizer, start_lr=1e-3, warmup_iter=40, end_iter=1000, num_iter=0)

    optim_manager = optim.OptimManager(loss_scale=2**20)
    optim_manager.add_optimizer(optimizer, lr_scheduler)
    pipe_rank = bmt.config["topology"].pipe_rank
    bmt.synchronize()
    avg_time_recorder = bmt.utils.AverageRecorder()
    avg_loss_recorder = bmt.utils.AverageRecorder()
    global_loss_items = []

    def forward_step(model, input, data):
        enc_input, pos, mask, targets = data
        input = model.preprocess_func((enc_input, pos)) if bmt.config["topology"].is_first_rank() else input[0]
        logits = model(input, pos, mask)
        if bmt.config["topology"].is_last_rank():
            logits = logits.view(-1, logits.shape[-1])
            targets = targets.view(-1)
            loss = loss_func(logits, targets)
            nonlocal global_loss_items
            global_loss = bmt.distributed.all_reduce(loss, comm=bmt.config["pp_tp_zero_comm"]).item()
            global_loss_items.append(global_loss)
            return loss, logits
        else:
            return logits
    
    def backward_step(output, grad_output):
        output = optim_manager.scale_loss(output)
        output = output / bmt.config['micros']
        torch.autograd.backward(output, grad_tensors=grad_output)
        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(bmt.config['load_stream'])

        
        
    for iteration in range(10):
        # load data
        st = time.time()
        rank = bmt.config["topology"].pipe_rank
        # global_loss, grad_norm = pipeline_forward_backward(model, data_loader(), micro , num_micros, optim_manager)
        pipeline_forward_backward(model, data_loader(), forward_step, backward_step, micro , num_micros)
        optim_manager.step()
        optim_manager.zero_grad()
        # record time and loss
        iteration_time = time.time() - st
   
        if bmt.config["topology"].is_last_rank():
            global_loss = sum(list(global_loss_items))/len(global_loss_items)
            avg_time_recorder.record(iteration_time)
            avg_loss_recorder.record(global_loss)
            print(
                "| Iter: {:6d} | loss: {:.10f} average_loss: {:.4f} | lr: {:.4e} scale: {:10.4f} | time: {:.4f}".format(
                    iteration,
                    global_loss,
                    avg_loss_recorder.value,
                    lr_scheduler.current_lr,
                    optim_manager.loss_scale,
                    avg_time_recorder.value
                )
            )

    

if __name__ == '__main__':
    main()
