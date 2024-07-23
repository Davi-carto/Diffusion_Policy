"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import argparse

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import os
import torch.multiprocessing as mp
import torch
from hydra.core.hydra_config import HydraConfig

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")
    parser.add_argument("--config_name", type=str, required=True, help="Name of the config file to use")
    return parser.parse_args()


def ddp_train(rank, world_size, config_path, config_name):
    """
    Args:
         rank: Unique identifier of each process
         world_size: Total number of processes
         cfg: Hydra config object
    """ 
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # 在初始化组进程之前，调用 set_device，它为每个进程设置默认 GPU。这对于防止 GPU:0 上的挂起或过度内存使用非常重要
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    @hydra.main(
    version_base=None,
    #pathlib读取train.py文件所在目录，使得config_path默认路径为./diffusion_policy/config目录
    #在Python中，__file__是一个内置变量，用于获取当前执行脚本的文件名。
    # 具体来说，它返回包含当前模块文件路径的字符串
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
    def main(cfg: OmegaConf,rank, world_size):
        # resolve immediately so all the ${now:} resolvers
        # will use the same time.
        OmegaConf.resolve(cfg)

        # Add rank and world_size to the config
        cfg.rank = rank
        cfg.world_size = world_size

        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg)
        workspace.run()

    main(rank=rank, world_size=world_size, config_name=config_name)

        # Destroy the process group
    torch.distributed.destroy_process_group() 

if __name__ == "__main__":
    args = get_args()
    mp.spawn(ddp_train, args=(args.world_size, args.config_name), nprocs=args.world_size)



#########先读取参数cfg的结构

"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import argparse

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import os
import torch.multiprocessing as mp
import torch
from hydra.core.hydra_config import HydraConfig

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")
    parser.add_argument("--output_dir", type=str, default='./data/outputs', help="Directory to save outputs")
    # parser.add_argument("--config_name", type=str, required=True, help="Name of the config file to use")
    return parser.parse_args()

@hydra.main(
    version_base=None,
    #pathlib读取train.py文件所在目录，使得config_path默认路径为./diffusion_policy/config目录
    #在Python中，__file__是一个内置变量，用于获取当前执行脚本的文件名。
    # 具体来说，它返回包含当前模块文件路径的字符串
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def load_config(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    return cfg

def main(cfg: OmegaConf,rank, world_size):

    # Add rank and world_size to the config
    cfg.rank = rank
    cfg.world_size = world_size

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg,output_dir=args.output_dir)
    workspace.run()

def ddp_train(rank, world_size, config_path, config_name):
    """
    Args:
         rank: Unique identifier of each process
         world_size: Total number of processes
         cfg: Hydra config object
    """ 
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # 在初始化组进程之前，调用 set_device，它为每个进程设置默认 GPU。这对于防止 GPU:0 上的挂起或过度内存使用非常重要
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    main(rank, world_size,)

        # Destroy the process group
    torch.distributed.destroy_process_group() 

if __name__ == "__main__":
    args = get_args()
    cfg = load_config()
    mp.spawn(ddp_train, args=(args.world_size,cfg), nprocs=args.world_size)




# ##### #3#####
# ##### #3#####
# ##### #3#####
# ##### #3#####

import sys
import torch
import hydra
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import os    

# Use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

# Allows arbitrary Python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

def ddp_setup(rank, world_size, cfg):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    cfg.rank = rank
    cfg.world_size = world_size

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg,output_dir=str(pathlib.Path(__file__).parent.joinpath('data', 'outputs', str(rank))))
    workspace.run()

    dist.destroy_process_group()

@hydra.main(
    version_base=None,
    # Pathlib reads the directory where train.py is located, making the default path for config_path './diffusion_policy/config'
    config_path=str(pathlib.Path(__file__).parent.joinpath('diffusion_policy', 'config'))
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    
    world_size = torch.cuda.device_count()
    mp.spawn(ddp_setup, args=(world_size, cfg,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()