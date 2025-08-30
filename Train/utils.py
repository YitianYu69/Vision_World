import torch
import torch.distributed as dist

import torchmetrics
from torchmetrics import Metric

import os
from typing import List, Union

def build_metrics(*, metric_lists: List[str],
                  task:str,
                  num_classes: int,
                  average_type: str,
                  sync: bool,
                  Union[str, torch.device] = "cpu"):
    kwargs = dict(task=task, num_classes=num_classes, average=average_type, sync_on_compute=sync)

    metrics = {
        'Accuracy' : Accuracy(**kwargs).to(device),
        'Recall' : Recall(**kwargs).to(device),
        'Precision' : Precision(**kwargs).to(device),
        'F1Score' : F1Score(**kwargs).to(device),
        'AUROC' : AUROC(**kwargs).to(device)
    }

    new_metrics = {}
    for m in metric_lists:
        if m not in metrics:
            raise ValueError(f"Current metric {m} is not supported! Please set it manually!")
        else:
            new_metrics[m] = metrics[m]
    return new_metrics
        

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
