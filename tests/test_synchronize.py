import torch
import bmtrain as bmt

from bmtrain.global_var import config
from bmtrain import nccl, distributed
from bmtrain.synchronize import gather_result

def test_main():
    tensor = torch.rand(5, 5)
    result = bmt.gather_result(tensor)
    
    tensor_slice_0 = tensor[:1, :1]
    result_slice_0 = bmt.gather_result(tensor_slice_0)

    tensor_slice_1 = tensor[:2, :2]
    result_slice_1 = bmt.gather_result(tensor_slice_1)

    tensor_slice_2 = tensor[:3, :3]
    result_slice_2 = bmt.gather_result(tensor_slice_2)

    tensor_slice_3 = tensor[:4, :4]
    result_slice_3 = bmt.gather_result(tensor_slice_3)

    print(result, result_slice_1, result_slice_2, result_slice_3, sep='\n')

if __name__ == '__main__':
    bmt.init_distributed(pipe_size=1)
    test_main()
