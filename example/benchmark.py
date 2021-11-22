import bmpretrain as bmp

def main():
    bmp.init_distributed()
    bmp.print_rank("======= All Gather =======")
    bmp.benchmark.all_gather()
    bmp.print_rank("===== Reduce Scatter =====")
    bmp.benchmark.reduce_scatter()
    

if __name__ == '__main__':
    main()