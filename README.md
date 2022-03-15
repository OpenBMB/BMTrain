<div align="center">

<h1>🚄 BMTrain</h1>

------

<p align="center">

<a href='https://bmtrain.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/bmtrain/badge/?version=latest' alt='Documentation Status' />
</a>

<a href="https://github.com/OpenBMB/BMTrain/releases">
    <img alt="GitHub release (latest by date including pre-releases)" src="https://img.shields.io/github/v/release/OpenBMB/BMTrain?include_prereleases">
</a>

<a href="https://github.com/OpenBMB/BMTrain/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/OpenBMB/BMTrain">
</a>

</p>

</div>


## 1. 安装

#### From PyPI (recommended)

```shell
$ pip install bmtrain
```

#### From source

```
$ git clone https://github.com/OpenBMB/BMTrain.git
$ cd BMTrain
$ python setup.py install
```

## 2. 使用

### Step 1: 启用 bmtrain

要使用bmtrain需要在代码中引入`bmtrain`工具包，并在代码的开头使用`bmtrain.init_distributed`

```python
import bmtrain as bmt
bmt.init_distributed(
    seed=0,
    # ...
)
```

**注意：** 使用`bmtrain`时请不要使用`pytorch`自带的`distributed`模块，包括`torch.distributed.init_process_group`以及相关通信函数。

### Step 2: 使用 ZeRO3 优化

使用ZeRO3优化需要对模型代码进行简单替换：

* `torch.nn.Module` -> `bmtrain.DistributedModule`
* `torch.nn.Parameter` -> `bmtrain.DistributedParameter`

并在合适的模块上使用`Checkpointing`。

**原始代码：**

```python
import torch
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.empty(1024))
        self.module_list = torch.nn.ModuleList([
            SomeTransformerBlock(),
            SomeTransformerBlock(),
            SomeTransformerBlock()
        ])
    
    def forward(self):
        x = self.param
        for module in self.module_list:
            x = module(x, 1, 2, 3)
        return x

```

**替换后代码：**

```python
import torch
import bmtrain as bmt
class MyModule(bmt.DistributedModule):
    def __init__(self):
        super().__init__()
        self.param = bmt.DistributedParameter(torch.empty(1024))
        self.module_list = torch.nn.ModuleList([
            bmt.CheckpointBlock(SomeTransformerBlock()),
            bmt.CheckpointBlock(SomeTransformerBlock()),
            bmt.CheckpointBlock(SomeTransformerBlock())
        ])
    
    def forward(self):
        x = self.param
        for module in self.module_list:
            x = module(x, 1, 2, 3)
        return x
    
```

### Step 3: 通信优化

为了进一步缩短通信额外开销，将通信与运算时间重叠，可以使用`TransformerBlockList`来进一步优化。
在使用时需要对代码进行简单替换：

* `torch.nn.ModuleList` -> `bmtrain.TransformerBlockList`
* `for module in self.module_list: x = module(x, ...)` -> `x = self.module_list(x, ...)`

**原始代码：**

```python
import torch
import bmtrain as bmt
class MyModule(bmt.DistributedModule):
    def __init__(self):
        super().__init__()
        self.param = bmt.DistributedParameter(torch.empty(1024))
        self.module_list = torch.nn.ModuleList([
            bmt.CheckpointBlock(SomeTransformerBlock()),
            bmt.CheckpointBlock(SomeTransformerBlock()),
            bmt.CheckpointBlock(SomeTransformerBlock())
        ])
    
    def forward(self):
        x = self.param
        for module in self.module_list:
            x = module(x, 1, 2, 3)
        return x
    
```

**替换后代码：**

```python
import torch
import bmtrain as bmt
class MyModule(bmt.DistributedModule):
    def __init__(self):
        super().__init__()
        self.param = bmt.DistributedParameter(torch.empty(1024))
        self.module_list = bmt.TransformerBlockList([
            bmt.CheckpointBlock(SomeTransformerBlock()),
            bmt.CheckpointBlock(SomeTransformerBlock()),
            bmt.CheckpointBlock(SomeTransformerBlock())
        ])
    
    def forward(self):
        x = self.param
        x = self.module_list(x, 1, 2, 3)
        return x
    
```

### Step 4: 运行分布式训练代码

bmtrain支持pytorch原生的分布式训练启动器：

#### torch.distributed.launch
```shell
$ python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node ${GPU_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} train.py
```

#### torchrun

```shell
$ torchrun --nnodes=${NNODES} --nproc_per_node=${GPU_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py
```

更多信息请参考pytorch官方文档：![Launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility)

## 3. 其它说明

`BMTrain`工具包对pytorch进行了底层修改，如果你的程序输出了意料之外的结果，可以在issue中提交相关信息。

更多例子请参考 *examples* 文件夹。

