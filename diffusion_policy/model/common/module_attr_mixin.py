import torch.nn as nn

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

# self.parameters()：
# self.parameters() 是 nn.Module 类的一个方法，返回一个生成器，生成模型中的所有参数。
# iter(self.parameters())：
# iter(self.parameters()) 将参数生成器转换为一个迭代器，这样我们就可以使用 next() 函数来获取第一个参数。
# next(iter(self.parameters()))：
# next(iter(self.parameters())) 获取迭代器中的第一个参数
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
