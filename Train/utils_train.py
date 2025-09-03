import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from torchmetrics import Metric, Accuracy, Recall, Precision, F1Score, AUROC

import math
from typing import List, Union, Callable

def warmup(model: nn.Module,
           criterion: Callable,
           dataset: Dataset, 
           sub_data_portion: float,
           device: str):
    data = Subset(dataset, len(dataset) * sub_data_portion)
    dl = DataLoader(data, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    model = model.to(device)

    for images, labels in dl:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()


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


class EMA():
    def __init__(self,
                 model,
                 decay = 0.999,
                 tau = 2000,
                 device = 'cpu',
                 use_fp32_master = True,
                 include_buffers = True,
                 param_filters = None,
                 update_every = 1):

        self.ema_model = self._unwrap_model(model)
        self.decay_base = float(decay)
        self.tau = int(tau)
        self.device = device
        self.use_fp32_master = use_fp32_master
        self.include_buffers = include_buffers
        self.param_filters = param_filters or (lambda n, p: p.requires_grad and p.dtype.is_floating_point)
        self.update_every = update_every
        self.num_updates = 0

        self.shadow_params = {}
        self.shadow_buffers = {}

        self._register(self.ema_model)

    def _unwrap_model(self, model):
        return model.module if hasattr(model,'module') else model
    
    def _register(self, model):
        self.shadow_params.clear()
        self.shadow_buffers.clear()

        for n, p in self._named_fparameters(model):
            ema_p = p.detach().clone()
            if self.use_fp32_master:
                ema_p = ema_p.to(torch.float32)
            if self.device != 'cpu' and self.device is not None:
                ema_p = ema_p.to(self.device)
            ema_p.requires_grad_(False)
            self.shadow_params[n] = ema_p

        if self.include_buffers:
            for n, b in model.named_buffers():
                ema_b = b.detach().clone()
                if self.device != 'cpu' and self.device is not None:
                    ema_b = ema_b.to(self.device)
                self.shadow_buffers[n] = ema_b
    
    def to(self, device):
        self.device = device
        for n in self.shadow_params.keys():
            self.shadow_params[n] = self.shadow_params[n].to(self.device)
        if self.include_buffers:
            for n in self.shadow_buffers.keys():
                self.shadow_buffers[n] = self.shadow_buffers[n].to(self.device)

    def update_parameters(self, model):
        if (self.num_updates + 1) % self.update_every != 0:
            self.num_updates += 1
            return

        model = self._unwrap_model(model)
        c_decay = self._current_decay()

        for n, p, ema_p in self._iter_named_fparameters_with_ema(model):
            if p is None or ema_p is None:
                continue
            
            src = p.detach()
            if self.use_fp32_master:
                upd = src.float()
            else:
                upd = src.to(dtype=ema_p.dtype)
            
            ema_p.mul_(c_decay).add_(upd, alpha=1 - c_decay)
        
        self.num_updates += 1

    # Overwrite model's weights with ema_model's weights for evaluation
    @torch.no_grad()
    def copy_to(self, model):
        model = self._unwrap_model(model)

        for n, p, ema_p in self._iter_named_fparameters_with_ema(model):
            if ema_p is not None:
                p.copy_(ema_p.to(device=p.device, dtype=p.dtype))
            
        if self.include_buffers:
            for n, b, ema_b in self._iter_named_buffers_with_ema(model):
                if b is not None and ema_b is not None:
                    b.copy_(ema_b.to(device=b.device, dtype=b.dtype))

    # Store the training model's weights for evaluation
    @torch.no_grad()
    def store(self, model):
        model = self._unwrap_model(model)

        self.backed_params = {n : p.detach().clone() for n, p in self._named_fparameters(model)}

        if self.include_buffers:
            self.backed_buffers = {n : b.detach().clone() for n, b in model.named_buffers()}
        else:
            self.backed_buffers = {}
    
    @torch.no_grad()
    def restore(self, model):
        model = self._unwrap_model(model)

        for n, p in self._named_fparameters(model):
            if n in self.backed_params:
                p.copy_(self.backed_params.get(n).to(device=p.device, dtype=p.dtype))
        
        if self.include_buffers:
            for n, b in model.named_buffers():
                if n in self.backed_buffers:
                    b.copy_(self.backed_buffers.get(n).to(device=b.device, dtype=b.dtype))


    def _current_decay(self):
        if self.tau and self.tau > 0:
            return 1 - (1 - self.decay_base) *  math.exp(-float(self.num_updates + 1) / float(self.tau))
        return self.decay_base

    def _named_fparameters(self, model):
        for n, p in model.named_parameters():
            if self.param_filters(n, p):
                yield(n, p)

    def _iter_named_fparameters_with_ema(self, model):
        for n, p in self._named_fparameters(model):
            ema_p = self.shadow_params.get(n, None)
            yield(n, p, ema_p)

    def _iter_named_buffers_with_ema(self, model):
        for n, b in model.named_buffers():
            ema_b = self.shadow_buffers.get(n, None)
            yield(n, b, ema_b)