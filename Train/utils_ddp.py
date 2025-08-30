import torch
import torch.distributed as dist

import os

def setup_ddp():
    dist.init_process_group(
        backend='nccl',
        init_method='env://'
    )

    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{local_rank}'

    torch.cuda.set_device(device)
    return local_rank, global_rank, world_size, device

def clean():
    dist.destory_process_group()
