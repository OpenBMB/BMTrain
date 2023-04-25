import bmtrain as bmt
import time
import torch
from statsd import get_statsd
import os

def main():
    node_name = os.getenv("NODE_NAME", "jeeves-hpc-gpu00")
    rank_name = (int)(os.getenv("RANK", "0")) // 8
    project_name = os.getenv("PROJECT_NAME", "no-project-name")
    log_name = "job-status.{}.{}.{}.step".format(
        project_name,
        rank_name,
        node_name,
    )
    statsd = get_statsd(log_name)

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
    statsd.gauges('calc_time', sum(time_lists) / len(time_lists))

if __name__ == '__main__':
    main()