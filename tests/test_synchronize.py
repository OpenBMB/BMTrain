import torch
import bmtrain as bmt

from bmtrain.global_var import config
from bmtrain import nccl, distributed
from bmtrain.synchronize import gather_result

def test_main():
    tensor = torch.rand(5, 5) * bmt.rank()
    result = bmt.gather_result(tensor)
    
    tensor_slice_0 = tensor[:1, :1]
    result_slice_0 = bmt.gather_result(tensor_slice_0)
    assert torch.allclose(result[:1, :1], result_slice_0, atol=1e-6), "Assertion failed for tensor_slice_0"

    tensor_slice_1 = tensor[:2, :2]
    result_slice_1 = bmt.gather_result(tensor_slice_1)
    assert torch.allclose(result[:2, :2], result_slice_1, atol=1e-6), "Assertion failed for tensor_slice_1"

    tensor_slice_2 = tensor[:3, :3]
    result_slice_2 = bmt.gather_result(tensor_slice_2)
    assert torch.allclose(result[:3, :3], result_slice_2, atol=1e-6), "Assertion failed for tensor_slice_2"

    tensor_slice_3 = tensor[:4, :4]
    result_slice_3 = bmt.gather_result(tensor_slice_3)
    assert torch.allclose(result[:4, :4], result_slice_3, atol=1e-6), "Assertion failed for tensor_slice_3"

    print("All slice tests passed!")

if __name__ == '__main__':
    bmt.init_distributed(pipe_size=1)
    test_main()
