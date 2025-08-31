import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchmetrics import Metric

import deepspeed

import time
from typing import Union, Callable, Dict, Optional

class Trainer():
    def __init__(self, 
                 *,
                 model: nn.Module,
                 compile_type: str = None,
                 DS_config: Dict = None,
                 DDP_config: Dict = None,
                 criterion: Union[nn.Module, Callable] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
                 scaler: torch.amp.GradScaler = None,
                 metrics: Dict[str, Metric],
                 ema: Callable = None,
                 num_epochs: int,
                 grad_acc_step: int = 1,
                 amp_enable: bool = True,
                 device: Union[str, torch.device] = 'cpu'):
        self.DS_config = DS_config
        self.DDP_config = DDP_config
        self.cri = criterion
        self.opt = optimizer
        self.sch = scheduler
        self.scaler = scaler
        self.metrics = metrics
        self.ema = ema
        self.num_epochs = num_epochs
        self.grad_acc_step = grad_acc_step
        self.amp_enable = amp_enable
        self.device = device
        
        # Avoid multiple dist function call
        rank0 = dist.get_rank() == 0

        # If using CUDA, move model to device first!
        if self.device != 'cpu':
            model.to(device)
        
        # Check for compile
        if compile_type is not None:
            fullgraph = False if DS_config is not None else True
            model.compile(fullgraph=fullgraph, mode=compile_type)
            if (DS_config is not None or DDP_config is not None) and compile_type != 'reduce-overhead' and rank0:
                logger.info("""For the max speed optimization, recommand to enable reduce-overhead compile mode
                            to avoid rebuild autograd graph!""")

        # ---------------------------------------------
        # Wrap model to enable different optimization
        # ---------------------------------------------
        if self.DS_config is not None:
            self.engine, _ = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                config=DS_config
            )
        elif self.DDP_config is not None:
            if dist.is_available() and dist.is_initialized() and rank0 and self.DDP_config['broadcast_buffers']:
                logger.info(f'Please turn off the broadcast_buffers, if you used the torch.nn.SyncBatchNorm.convert_sycn_batchnorm()')
            if dist.is_available() and dist.is_initialized() and rank0 and self.DDP_config['gradient_as_bucket_view']:
                logger.info(f'Please turn on the set_to_none for your optimizaer.zer_grad()! If you do not, then your DDP grad bucket will be zeroed out!')

            model.to(device)

            ddp_kwargs = dict(
                static_graph=self.DDP_config.get('static_graph', False),
                broadcast_buffers=self.DDP_config.get('broadcast_buffers', True),
                bucket_cap_mb=self.DDP_config.get('bucket_cap_mb', 25),
                find_unused_parameters=self.DDP_config.get('find_unused_parameters', False),
                gradient_as_bucket_view=self.DDP_config.get('gradient_as_bucket_view', False),
            )

            self.engine = DDP(
                model,
                device_ids=self.DDP_config.get('device_ids'),
                **ddp_kwargs)
        else:
            self.engine = model

        # ---------------------------------------------------
        # If TP2 AMP enabled, auto check the best cast dtype
        # ---------------------------------------------------
        if not self._is_deepspeed() and amp_enable:
            major, _ = torch.cuda.get_device_capability(torch.device(device))
            self.cast_dtype = torch.bfloat16 if major >= 8 and torch.cuda.is_available() else torch.float16

            if self.cast_dtype == torch.float16 and scaler is None:
                raise ValueError(f"AMP float16 is enabled, then the scaler cannot be None!")
        else:
            self.cast_dtype = None

    def _is_deepspeed(self):
        return isinstance(self.engine, deepspeed.DeepSpeedEngine)

    def _guard_all_reduce_SUM(self, t):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t

    def _training_step(self, data, target):
        if self._is_deepspeed():
            logits = self.engine(data)
            loss = self.cri(logits, target)
            self.engine.backward(loss)
            self.engine.step()
        else:
            self.opt.zero_grad(set_to_none=True)
            device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
            with torch.autocast(device_type=device_type, dtype=self.cast_dtype, enabled=self.amp_enable):
                logits = self.engine(data)
                loss = self.cri(logits, target)
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.engine.parameters(), max_norm=1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.engine.parameters(), max_norm=1.0)
                self.opt.step()
        
        if self.ema is not None:
            self.ema.update_parameters(self.engine)
        return logits, loss

    @torch.no_grad()
    def _update_metrics(self, logits, target):
        for k, v in self.metrics.items():
            if k == 'AUROC':
                v.update(F.softmax(logits, dim=1), target)
            else:
                v.update(logits.argmax(dim=1), target)


    def _training_ds(self,
                 dataloader: DataLoader,
                 epoch: int):
        total_loss, data_len = torch.tensor(0.0, dtype=torch.float32, device=self.device), torch.tensor(0, dtype=torch.long, device=self.device)
        computed_metrics = {}

        is_wrapped = isinstance(self.engine, (DDP, deepspeed.DeepSpeedEngine))
        (self.engine.module if is_wrapped else self.engine).train()

        for v in self.metrics.values():
            v.reset()

        if isinstance(dataloader.sampler, DistributedSampler) and self.DDP_config is not None:
            dataloader.sampler.set_epoch(epoch)

        start_time = time.time()
        for data, target in dataloader:
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

            logits, loss = self._training_step(data, target)
            self._update_metrics(logits, target)
        
            total_loss += loss.detach() * data.size(0)
            data_len += len(data)

        end_time = time.time()
        total_loss = self._guard_all_reduce_SUM(total_loss)
        data_len = self._guard_all_reduce_SUM(data_len)
        computed_metrics['Loss'] = total_loss.item() / data_len.item()
        computed_metrics['Time'] = end_time - start_time
        for k, v in self.metrics.items():
            computed_metrics[k] = v.compute()
        
        return computed_metrics

    @torch.no_grad()
    def _validation(self,
                    dataloader: DataLoader,
                    epoch: int):
        total_loss, data_len = torch.tensor(0.0, dtype=torch.float32, device=self.device), torch.tensor(0, dtype=torch.long, device=self.device)
        computed_metrics = {}

        is_wrapped = isinstance(self.engine, (DDP, deepspeed.DeepSpeedEngine))
        (self.engine.module if is_wrapped else self.engine).eval()

        for v in self.metrics.values():
            v.reset()

        start_time = time.time()
        for data, target in dataloader:
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

            logits = self.engine(data)
            loss = self.cri(logits, target)

            self._update_metrics(logits, target)
            total_loss += loss.detach() * data.size(0)
            data_len += len(data)
            
        
        end_time = time.time()
        total_loss = self._guard_all_reduce_SUM(total_loss)
        data_len = self._guard_all_reduce_SUM(data_len)
        computed_metrics['Loss'] = total_loss.item() / data_len.item()
        computed_metrics['Time'] = end_time - start_time
        for k, v in self.metrics.items():
            computed_metrics[k] = v.compute()
        return computed_metrics

    
