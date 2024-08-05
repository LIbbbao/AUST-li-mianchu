# modeling/models.py



from .transformer import TransformerEncoder
from .saits import SAITS  # 假设 SAITS 模型在此处定义

MODEL_DICT = {
    'Transformer': TransformerEncoder,
    'SAITS': SAITS,
    # 其他模型定义...
}
import torch
import torch.nn as nn

class ExampleModel(nn.Module):
    def __init__(self, **kwargs):
        super(ExampleModel, self).__init__()
        # 初始化模型参数
        pass

    def forward(self, x):
        # 定义前向传播
        return x

# 定义模型字典
MODEL_DICT = {
    'ExampleModel': ExampleModel,
    # 添加其他模型类
}

# 定义优化器字典
OPTIMIZER = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    # 添加其他优化器
}

