import torch
from torchmetrics import Metric, Accuracy, Recall, Precision, F1Score, AUROC

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