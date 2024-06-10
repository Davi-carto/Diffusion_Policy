from typing import Dict

import torch
import torch.nn
from diffusion_policy.model.common.normalizer import LinearNormalizer

class BaseLowdimDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseLowdimDataset':
        # return an empty dataset by default
        return BaseLowdimDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: T, Do
            action: T, Da
        """
        raise NotImplementedError()


class BaseImageDataset(torch.utils.data.Dataset):
    # def get_validation_dataset(self) -> 'BaseLowdimDataset':
    ##上面是原来的，->指定的类型为什么是BaseLowdimDataset，而不是BaseImageDataset？
    ##是写错了？return BaseImageDataset()和->'BaseImageDataset'谁的优先级更高？
    def get_validation_dataset(self) -> 'BaseImageDataset':
        # return an empty dataset by default
        return BaseImageDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()

    # 常见的特殊方法（魔术方法）包括
    # __init__（类的构造函数）、__str__（定制对象的打印输出）、__len__（返回对象的长度）、
    # __getitem__（允许对象按索引访问元素）等。
    # 当你调用内置函数或执行特定操作时，Python
    # 会自动调用这些特殊方法，从而实现相应的功能，例如创建对象、获取对象长度、比较对象、迭代对象等。

    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()
