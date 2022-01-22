# 使用方法

## 安装

```
python3 setup.py install
```

## 使用

使用时需要对原有代码进行简单替换：

**原始代码：**

```python
import torch
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.empty(1024))
        self.module_list = torch.nn.ModuleList([
            torch.nn.Linear(1024, 1024),
            torch.nn.Linear(1024, 1024),
            torch.nn.Linear(1024, 1024)
        ])
    
    def forward(self):
        x = self.param
        for module in self.module_list:
            x = module(x)
        return x

```

**替换后代码：**

```python
import torch
import bmpretrain as bmp
class MyModule(bmp.DistributedModule):
    def __init__(self):
        super().__init__()
        self.param = bmp.DistributedParameter(torch.empty(1024))
        self.module_list = bmp.TransformerBlockList([
            bmp.CheckpointBlock(MyModule()),
            bmp.CheckpointBlock(MyModule()),
            bmp.CheckpointBlock(MyModule())
        ])
    
    def forward(self):
        x = self.param
        x = self.module_list(x)
        return x
    
```

## 说明

在使用时需要将所有的`torch.nn.Module`替换为`bmp.DistributedModule`，将`torch.nn.Parameter`替换为`bmp.DistributedParameter`。

同时，针对于**transformer**等多层堆叠的模型，可以使用`bmp.TransformerBlockList`和`bmp.CheckpointBlock`来实现`checkpointing`和进一步的加速。

# 其他

使用`bmpretrain.inspect`模块来方便的打印模型参数分布，和中间变量分布。

更多例子请参考 *examples* 文件夹。

