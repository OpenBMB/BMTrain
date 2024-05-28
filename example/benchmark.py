import bmtrain as bmt
from bmtrain import benchmark

def main():
    bmt.init_distributed()
    bmt.print_rank("======= All Gather =======")
    benchmark.all_gather()
    bmt.print_rank("===== Reduce Scatter =====")
    benchmark.reduce_scatter()
    bmt.print_rank("===== All 2 All =====")
    benchmark.all2all()
    bmt.print_rank("===== All 2 One =====")
    benchmark.all2one()

if __name__ == '__main__':
    main()
