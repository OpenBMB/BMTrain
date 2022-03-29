<div align="center">

<h1>🚄 BMTrain</h1>

**大模型高效训练工具包**

<p align="center">
  <a href="#总览">总览</a> • <a href="#文档">文档</a> • <a href="#安装">安装</a> • <a href="#使用说明">使用说明</a> • <a href="#性能">性能</a> • <a href="./README.md" target="_blank">English</a>
<br>
</p>

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

## 最新动态
- 2022/03/16 [0.1.1](https://github.com/OpenBMB/BMTrain/releases/tag/0.1.1) BMTrain 公开发布了第一个稳定版本，修复了beta版本中的一些问题.
- 2022/02/11 [0.0.15](https://github.com/OpenBMB/BMTrain/releases/tag/0.0.15) BMTrain 公开发布了第一个 beta 版本。

<div id="总览"></div>

## 总览

BMTrain 是一个高效的大模型训练工具包，可以用于训练数百亿参数的大模型。BMTrain 可以在分布式训练模型的同时，能够保持代码的简洁性。

<div id="文档"></div>

## 文档
我们的[文档](https://bmtrain.readthedocs.io/en/latest/index.html) 提供了关于工具包的更多信息。

<div id="安装"></div>

## 安装

- 用 pip 安装（推荐）: ``pip install bmtrain``

- 从源代码安装: 下载工具包，然后运行 ``python setup.py install``

安装 BMTrain 可能需要花费数分钟的时间，因为在安装时需要编译 c/cuda 源代码。
我们推荐直接在训练环境中编译 BMTrain，以避免不同环境带来的潜在问题。


<div id="使用说明"></div>

## 说明

### 步骤 1: 启用 BMTrain

首先，你需要在代码开头初始化 BMTrain。正如在使用 PyTorch 的分布式训练模块需要在代码开头使用 **init_process_group**一样，使用 BMTrain 需要在代码开头使用 **init_distributed**。

```python
import bmtrain as bmt
bmt.init_distributed(
    seed=0,
    # ...
)
```

**注意：** 使用 BMTrain 时请不要使用 PyTorch 自带的 `distributed` 模块，包括 `torch.distributed.init_process_group` 以及相关通信函数。

### 步骤 2: 使用 ZeRO-3 优化

使用ZeRO3优化需要对模型代码进行简单替换：

* `torch.nn.Module` -> `bmtrain.DistributedModule`
* `torch.nn.Parameter` -> `bmtrain.DistributedParameter`

And wrap the transformer blocks with `bmtrain.CheckpointBlock`.
并在 transformer 模块上使用 `bmtrain.CheckpointBlock`。

下面是一个例子：

**原始代码**

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

**替换后代码**

```python
import torch
import bmtrain as bmt
class MyModule(bmt.DistributedModule): # 修改这里
    def __init__(self):
        super().__init__()
        self.param = bmt.DistributedParameter(torch.empty(1024)) # 修改这里
        self.module_list = torch.nn.ModuleList([
            bmt.CheckpointBlock(SomeTransformerBlock()), # 修改这里
            bmt.CheckpointBlock(SomeTransformerBlock()), # 修改这里
            bmt.CheckpointBlock(SomeTransformerBlock())  # 修改这里
        ])
    
    def forward(self):
        x = self.param
        for module in self.module_list:
            x = module(x, 1, 2, 3)
        return x
    
```

### 步骤 3: 通信优化

为了进一步缩短通信额外开销，将通信与运算时间重叠，可以使用`TransformerBlockList`来进一步优化。

在使用时需要对代码进行简单替换：

* `torch.nn.ModuleList` -> `bmtrain.TransformerBlockList`
* `for module in self.module_list: x = module(x, ...)` -> `x = self.module_list(x, ...)`

**原始代码**

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

**替换后代码**

```python
import torch
import bmtrain as bmt
class MyModule(bmt.DistributedModule):
    def __init__(self):
        super().__init__()
        self.param = bmt.DistributedParameter(torch.empty(1024))
        self.module_list = bmt.TransformerBlockList([ # 修改这里
            bmt.CheckpointBlock(SomeTransformerBlock()),
            bmt.CheckpointBlock(SomeTransformerBlock()),
            bmt.CheckpointBlock(SomeTransformerBlock())
        ])
    
    def forward(self):
        x = self.param
        x = self.module_list(x, 1, 2, 3) # 修改这里
        return x
    
```

### 步骤 4: 运行分布式训练代码

BMTrain 使用 PyTorch 原生的分布式训练启动器，你可以根据 PyTorch 版本选择下列命令中的一个。

* `${MASTER_ADDR}` 为主节点的 IP 地址
* `${MASTER_PORT}` 为主节点的端口
* `${NNODES}` 为节点数量
* `${GPU_PER_NODE}` 为每个节点的 GPU 数量
* `${NODE_RANK}` 为本节点的 rank

#### torch.distributed.launch
```shell
$ python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node ${GPU_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} train.py
```

#### torchrun

```shell
$ torchrun --nnodes=${NNODES} --nproc_per_node=${GPU_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py
```

更多信息请参考 PyTorch [官方文档](https://pytorch.org/docs/stable/distributed.html#launch-utility)。

## 样例

我们提供了一个使用 BMTrain 训练 GPT-2 的[样例](https://github.com/OpenBMB/BMTrain/tree/main/example)。
代码主要包含以下几个部分。

### 第 1 部分: 模型定义

```
├── layers
│   ├── attention.py
│   ├── embedding.py
│   ├── feedforward.py
│   ├── __init__.py
│   ├── layernorm.py
│   └── linear.py
└── models
    ├── gpt.py
    └── __init__.py
```

上面是代码的目录结构。

我们定义了 GPT-2 需要的所有模型层，并使用 BMTrain 的 `DistributedModule` 和 `DistributedParameter` 来启用 ZeRO-3 优化。

### 第 2 部分: 初始化 BMTrain

```python
bmtrain.init_distributed(seed=0)

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

bmtrain.init_parameters(model) # 或者使用`bmtrain.load`加载checkpoint

# ... 其他初始化（例如数据集） ...
```

`bmtrain.init_distributed(seed=0)` 用于初始化分布式训练环境，并设置随机数种子便于复现。

`bmtrain.init_parameters(model)` 用于初始化模型的分布式参数。

### 第 3 部分: 初始化优化器和学习率调整策略

```python
loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
optimizer = bmtrain.optim.AdamOffloadOptimizer(model.parameters(), weight_decay=1e-2, scale=2**20)
lr_scheduler = bmtrain.lr_scheduler.Noam(optimizer, start_lr=1e-3, warmup_iter=40, end_iter=1000, num_iter=0)
```

BMTrain 支持**所有** PyTorch 原生的优化器和损失函数，同时你也可以使用 BMTrain 提供的融合（fused）优化器用于混合精度训练。

此外，在 `bmtrain.lr_scheduler` 中 BMTrain 也提供了常见的学习率调整策略。

### 第 4 部分: 训练

```python
for iteration in range(1000):
    # ... 为每个rank加载数据 ...

    # 梯度清零
    optimizer.zero_grad()

    # 前向传播
    pos = torch.arange(enc_input.size(1)).long().cuda().repeat(enc_input.size(0), 1)
    logits = model(
        enc_input,
        pos,
        pos < enc_length[:, None]
    )
    batch, seq_len, vocab_out_size = logits.size()

    loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))
    
    global_loss = bmtrain.sum_loss(loss).item() # 聚合所有rank上的损失

    # 损失缩放和反向传播
    loss = optimizer.loss_scale(loss)
    loss.backward()

    # 更新参数
    bmtrain.optim_step(optimizer, lr_scheduler)

    # ... 保存checkpoint、打印日志 ...
```

这部分代码略有些长，但写起来就像常见的训练代码一样，你不需要为分布式训练调整太多的代码。

你可以根据代码中的注释来了解各部分代码的作用。

唯一需要说明的是 `optimizer.loss_scale`，损失缩放是混合精度训练中的一项常用技术，用于避免梯度下溢。如果你没有使用 BMTrain 中的融合优化器，你可以删除这行代码。

<div id="性能"></div>

## 性能

我们训练了一个有130亿参数的 GPT-2 模型，使用了4台服务器，每台服务器有8张V100显卡。我们测试了训练过程中每个GPU的吞吐量（每个GPU每秒处理的样本数），结果见下表。

模型结构：
* 40层
* 128个注意力头
* 5120的隐藏层维数
* 512的序列长度


| batch size  | 8     | 16    | 24    | 32    |
|-------------|-------|-------|:------|:------|
| BMTrain     | 24.15 | 26.94 | 29.42 | 28.28 |
| ZeRO3(mp=1) | 14.88 | 21.69 | 24.38 | -     |
| ZeRO3(mp=4) | 15.51 | -     | -     | -     |
| ZeRO3(mp=8) | 15.51 | -     | -     | -     |
| ZeRO2(mp=1) | -     | -     | -     | -     |
| ZeRO2(mp=4) | 22.85 | -     | -     | -     |
| ZeRO2(mp=8) | 21.33 | -     | -     | -     |

**ZeROa(mp=b)** 表示 DeepSpeed + Megatron ZeRO stage a 和 model parallelism = b。

表格中 **-** 表示超出显存。

## 模型支持

我们已经将大多数常见的 NLP 模型移植到了 BMTrain 中。你可以在 [ModelCenter](https://github.com/OpenBMB/ModelCenter) 项目中找到支持模型的列表。

## 开源社区
欢迎贡献者参照我们的[贡献指南](https://github.com/OpenBMB/BMTrain/blob/master/CONTRIBUTING.md)贡献相关代码。

您也可以在其他平台与我们沟通交流：
- QQ群: 735930538
- 官方网站: http://www.openbmb.org
- 微博: http://weibo.cn/OpenBMB
- Twitter: https://twitter.com/OpenBMB

## 开源许可

该工具包使用[Apache 2.0](https://github.com/OpenBMB/BMTrain/blob/main/LICENSE)开源许可证。

## 其他说明

`BMTrain` 工具包对 PyTorch 进行了底层修改，如果你的程序输出了意料之外的结果，可以在issue中提交相关信息。
