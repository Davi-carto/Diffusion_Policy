from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class PushTImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None##这个参数的作用？
            ):
        
        super().__init__()
        ##replay_buffer的作用？
        ##读取zarr文件，repaly_buffer.root可当dict使用，里面储存了数据
        # replay_buffer主体基于zarr,自定义了data和meta两个group管理数组，最后一级为数组名，可按Dict读写
        # -data：
        #           -img(time,H,W,3)
        #           -state(time,5)
        #           -action(time,2)
        # -meta：
        #           -episode_ends(n_episodes): 1维数组，内容为每个episode的time end index
        # ###time为所有episodes的总时长

        ###问题:这里调用了replay_buffer.copy_from_path()，没有传入store参数，那是按照numpy_backend处理么？
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])


        # 对数据集进行验证集和训练集的划分，并确保训练集不超过指定的最大训练集数目。
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,###什么情况需要pad？
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:2]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

###sample_to_data()函数用来实现接口规范
    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        # np.moveaxis(sample['img'], -1, 1)：这个操作通过移动数组的轴来重新排列数组的维度。在这里，-1代表最后一个维度，1代表我们希望将该维度移动到的位置。
        # 这个操作的目的是将图像数组的通道维度移动到第二个位置，以适应PyTorch对图像数据的要求
        # / 255：将数组中的所有元素除以255，这是一种常见的归一化操作，将图像的像素值缩放到0到1之间
        image = np.moveaxis(sample['img'],-1,1)/255
        # 这里T为一个horizon的长度
        #为什么没有截取成成T_o和T_a？
        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 2     仿真环境中只有x，y两个维度的动作
        }
        return data

#torch.utils.data.Dataset是代表自定义数据集方法的类，用户可以通过继承该类来自定义自己的数据集类，
    # 在继承时要求用户重载__len__()和__getitem__()这两个魔法方法
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        ###这里的数据是会被送入模型训练的一个样本
        ###因此应满足，T_o长度的Observations，T_a长度的Actions
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)

        return torch_data

#Python 中的魔法方法是指具有特殊名称和双下划线开头和结尾的方法，例如 __init__、__call__ 等。这些方法允许开发者在类中定制对象的行为，例如初始化对象、进行比较操作、使对象可调用等。魔法方法是 Python 面向对象编程中的重要概念，通过使用这些特殊方法，开发者可以实现自定义对象的特定行为和操作。

# 下面是一些常用的 Python 魔法方法的简要介绍：
#
# __init__(self, ...): 初始化方法，用于创建对象时进行初始化操作。
#
# __str__(self): 字符串表示方法，当使用 str(obj) 或 print(obj) 时调用，返回对象的字符串表示形式。
#
# __repr__(self): 对象表示方法，当使用 repr(obj) 时调用，返回对象的官方字符串表示形式。
#
# __call__(self, ...): 可调用方法，使得对象可以被调用，类似函数的行为。
#
# __getitem__(self, key): 获取元素方法，使得对象可以像序列或映射一样使用索引或键来获取元素。
#
# __len__(self): 长度方法，返回对象的长度，使得对象可以像序列一样使用 len() 函数。
#
# __add__(self, other): 加法方法，定义对象的加法行为。
#
# __eq__(self, other): 等于方法，定义对象的相等性比较行为。

def test():
    import os
    zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
