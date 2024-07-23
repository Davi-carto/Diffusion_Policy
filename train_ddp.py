import sys
import torch
import hydra
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import os   
import argparse

from hydra.core.hydra_config import HydraConfig
# Use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

# Allows arbitrary Python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")
    parser.add_argument("--config_path", type=str, default="./diffusion_policy/config", help="Path to the config directory")
    parser.add_argument("--config_name", type=str, default="train_diffusion_unet_real_image_workspace_ddp", help="Name of the config file to use")
    return parser.parse_args()

def ddp_setup(rank, world_size, cfg):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    cfg.rank = rank
    cfg.world_size = world_size

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

    dist.destroy_process_group()

def main(rank, world_size, config_name, config_path):
    with hydra.initialize(version_base=None, config_path=config_path, job_name=str(rank)):
        cfg = hydra.compose(config_name=config_name)
        OmegaConf.resolve(cfg)
        # OmegaConf.set_struct(cfg, True)
                # Manually set HydraConfig
        HydraConfig.instance().set_config(cfg)
        ddp_setup(rank, world_size, cfg)

if __name__ == "__main__":
    args = get_args()
    config_path = args.config_path
    config_name = args.config_name
    world_size = args.world_size
    mp.spawn(main, args=(world_size, config_name, config_path), nprocs=world_size, join=True)