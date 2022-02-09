# 使用方法

## 安装

```
python3 setup.py install
```

## 使用

### Step 1: ZeRO3

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

### Step 2: 通信优化

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

# 注意

请使用`bmtrain.inspect`模块来访问模型的参数和运算的中间变量以获取正确的结果。

更多例子请参考 *examples* 文件夹。

