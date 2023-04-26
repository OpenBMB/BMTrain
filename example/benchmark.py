import bmtrain as bmt
import time
import torch
from statsd import get_statsd
import os

def main():
    rank_name = (int)(os.getenv("RANK", "0")) // 8
    statsd = get_statsd()

    bmt.init_distributed()
    for i in range(10):
        bmt.print_rank("======= All Gather =======")
        bmt.benchmark.all_gather()
        bmt.print_rank("===== Reduce Scatter =====")
        bmt.benchmark.reduce_scatter()

    time_lists = []
    a = torch.ones((4096*16, 4096*64)).cuda()
    b = torch.ones((512, 4096*16)).cuda()
    for i in range(10):
        time_1 = time.time()
        res = torch.mm(b, a)
        torch.cuda.synchronize() 
        time_2 = time.time()
        time_lists.append(time_2 - time_1)
    statsd.gauges('calc_time', sum(time_lists) / len(time_lists), repeat=10)
    print (sum(time_lists) / len(time_lists), rank_name)

if __name__ == '__main__':
    main()