import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchmetrics import Metric

import deepspeed

import logging
from utils_train import warmup, build_CUDA_Graph
from utils_ddp import check_ddp

import time
from typing import Union, Callable, Dict, Optional

logger = logging.getLogger(__name__)

class Trainer():
    def __init__(self, 
                 *,
                 model: nn.Module,
                 teacher_model: nn.Module = None,
                 compile_type: str = None,
                 DS_config: Dict = None,
                 DDP_config: Dict = None,
                 CUDA_Graph: bool = False,
                 dataloader: DataLoader = None,
                 sub_data_portion: float = 1.0,
                 criterion: Union[nn.Module, Callable] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
                 scaler: torch.amp.GradScaler = None,
                 metrics: Dict[str, Metric] = None,
                 ema: Callable = None,
                 num_epochs: int = 200,
                 grad_acc_step: int = 1,
                 amp_enable: bool = True,
                 device: Union[str, torch.device] = 'cpu'):
        self.DS_config = DS_config
        self.DDP_config = DDP_config
        self.CUDA_Graph = CUDA_Graph
        self.train_dataloader = dataloader
        self.cri = criterion
        self.opt = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.metrics = metrics
        self.ema = ema
        self.num_epochs = num_epochs
        self.grad_acc_step = grad_acc_step
        self.amp_enable = amp_enable
        self.device = device.type if isinstance(device, torch.device) else device
        self.teacher_model = teacher_model
        
        # Avoid multiple dist function call
        self.rank0 = (dist.is_available() and dist.is_initialized() and dist.get_rank() == 0)

        # If using CUDA, move model to device first!
        if self.device != 'cpu':
            model.to(device)
            if teacher_model is not None:
                teacher_model.to(device).eval()
                for p in teacher_model.parameters():
                    p.requires_grad = False
                self.teacher_model = teacher_model
        
        # Check for compile
        if compile_type is not None:
            fullgraph = False if self.DS_config is not None else True
            model.compile(fullgraph=fullgraph, mode=compile_type)
            if (self.DS_config is not None or self.DDP_config is not None) and compile_type != 'reduce-overhead' and self.rank0:
                logger.info("""For the max speed optimization, consider enabling reduce-overhead compile mode
                            to avoid rebuilding autograd graph!""")
            if teacher_model is not None:
                teacher_model.compile(fullgraph=fullgraph, mode=compile_type)
                self.teacher_model = teacher_model


        # ---------------------------------------------------
        # If TP2 AMP enabled, auto check the best cast dtype
        # ---------------------------------------------------
        if self.DS_config is None and amp_enable and self.device.startswith('cuda'):
            major, _ = torch.cuda.get_device_capability(torch.device(device))
            self.cast_dtype = torch.bfloat16 if major >= 8 and torch.cuda.is_available() else torch.float16

            if self.cast_dtype == torch.float16 and scaler is None:
                raise ValueError(f"AMP float16 is enabled, then the scaler cannot be None!")
        elif self.DS_config is None and amp_enable and self.device.startswith('cpu'):
            self.cast_dtype = torch.bfloat16
        else:
            self.cast_dtype = None
            

        # ---------------------------------------------
        # Wrap model to enable different optimization
        # ---------------------------------------------
        if self.CUDA_Graph and self.DS_config is None:
            logger.info("CUDA Graph Enabled!")
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.Stream(side):
                self.engine = self._wrap_model_to_engine(model)
            logger.info("CUDA Graph Warmup!")
            warmup(self.engine, self.cri, self.train_dataloader.dataset, sub_data_portion, device)
            (self.graph_sync, self.graph_no_sync, self.static_x, self.static_y, self.static_logits, self.static_loss, self.compute_stream) = build_CUDA_Graph(self.engine, 
                                                                                                                                            self.cri, self.train_dataloader, 
                                                                                                                                            self.amp_enable, self.cast_dtype, 
                                                                                                                                            self.device, self.scaler, self.grad_acc_step)
            self.copy_stream = torch.cuda.Stream(device=device)
            self.copy_event = torch.cuda.Event()
        else:
            self.engine = self._wrap_model_to_engine(model)


    def train(self, epoch):
        return self._training(epoch)
    def valid(self, dataloader):
        return self._validation(dataloader)

    def _is_deepspeed(self):
        return isinstance(self.engine, deepspeed.DeepSpeedEngine)

    def _guard_all_reduce_SUM(self, t):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t

    def _wrap_model_to_engine(self, model, wrap_type='raw'):
        model.to(self.device)
        if self.DS_config is not None:
            engine, _ = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                config=self.DS_config
            )
            logger.info("Model Wrap Type: DeepSpeed!")
        elif self.DDP_config is not None:
            if check_ddp() and self.rank0 and self.DDP_config.get('broadcast_buffers', True):
                logger.info('Please turn off the broadcast_buffers if you used the torch.nn.SyncBatchNorm.convert_sync_batchnorm().')
            if check_ddp() and self.rank0 and self.DDP_config.get('gradient_as_bucket_view', False):
                logger.info('Please set set_to_none=True for optimizer.zero_grad(); otherwise DDP grad buckets may be zeroed out.')

            ddp_kwargs = dict(
                static_graph=self.DDP_config.get('static_graph', False),
                broadcast_buffers=self.DDP_config.get('broadcast_buffers', True),
                bucket_cap_mb=self.DDP_config.get('bucket_cap_mb', 25),
                find_unused_parameters=self.DDP_config.get('find_unused_parameters', False),
                gradient_as_bucket_view=self.DDP_config.get('gradient_as_bucket_view', False),
            )

            engine = DDP(
                model,
                device_ids=self.DDP_config.get('device_ids'),
                **ddp_kwargs)
            logger.info("Model Wrap Type: DDP")
        else:
            engine = model
            logger.info("Model Wrap Type: Raw")

        return engine

    def _training_step(self, data, target, grad_step):
        if self.teacher_model is not None:
            with torch.inference_mode():
                teacher_logits = self.teacher_model(data)

        if self._is_deepspeed():
            logits = self.engine(data)
            ori_loss = self.cri(logits, labels=target) if self.teacher_model is None else self.cri(logits, labels=target, teacher_logits=teacher_logits)
            self.engine.backward(ori_loss)
            self.engine.step()
        else:
            # Branch using CUDA Graph or not
            if not self.CUDA_Graph:
                device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
                with torch.autocast(device_type=device_type, 
                                    dtype=(self.cast_dtype if device_type in ['cuda', 'cpu'] else None),
                                    enabled=self.amp_enable and device_type in ['cuda', 'cpu']):
                    logits = self.engine(data)
                    ori_loss = self.cri(logits, labels=target) if self.teacher_model is None else self.cri(logits, labels=target, teacher_logits=teacher_logits)
                    
                    if isinstance(self.engine, nn.parallel.DistributedDataParallel) and self.grad_acc_step > 1:
                        self.engine.require_backward_grad_sync = grad_step
                    loss = ori_loss / self.grad_acc_step
                    if self.amp_enable and self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
            else:
                if data.shape != self.static_x.shape or target.shape != self.static_y.shape:
                    raise RuntimeError(
                        f"CUDA Graph expects fixed shapes. "
                        f"Got data {tuple(data.shape)} vs {tuple(self.static_x.shape)}, "
                        f"target {tuple(target.shape)} vs {tuple(self.static_y.shape)}."
                    )

                with torch.cuda.stream(self.copy_stream):
                    self.static_x.copy_(data, non_blocking=True)
                    self.cri.set_batch_target(grad_acc_step=self.grad_acc_step, labels=target,) if self.teacher_model is None else self.cri.set_batch_target(grad_acc_step=self.grad_acc_step, labels=target, teacher_logits=teacher_logits)
                    self.copy_event.record(self.copy_stream)

                with torch.cuda.stream(self.compute_stream):
                    self.compute_stream.wait_event(self.copy_event)
                    if grad_step:
                        self.graph_sync.replay()
                    else:
                        self.graph_no_sync.replay()

                logits = self.static_logits
                ori_loss = self.static_loss
            
            if grad_step:
                if self.amp_enable and self.scaler is not None:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.engine.parameters(), max_norm=1.0)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                else:
                    torch.nn.utils.clip_grad_norm_(self.engine.parameters(), max_norm=1.0)
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)

        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.update_parameters(self.engine)
        return logits, ori_loss

    @torch.no_grad()
    def _update_metrics(self, logits, target):
        for k, v in self.metrics.items():
            if k == 'AUROC':
                if logits.ndim == 1 or (logits.ndim == 2 and logits.size(1) == 1):
                    v.update(torch.sigmoid(logits), target)
                else:
                    v.update(F.softmax(logits, dim=1), target)
            else:
                v.update(logits.argmax(dim=1), target)


    def _training(self,
                 epoch: int):
        total_loss, data_len = torch.tensor(0.0, dtype=torch.float32, device=self.device), torch.tensor(0, dtype=torch.long, device=self.device)
        computed_metrics = {}

        is_wrapped = isinstance(self.engine, (DDP, deepspeed.DeepSpeedEngine))
        (self.engine.module if is_wrapped else self.engine).train()

        for v in self.metrics.values():
            v.reset()

        if isinstance(self.train_dataloader.sampler, DistributedSampler):
            self.train_dataloader.sampler.set_epoch(epoch)

        start_time = time.time()
        for step, (data, target) in enumerate(self.train_dataloader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            grad_step = ((step + 1) % self.grad_acc_step == 0 or (step + 1) == len(self.train_dataloader))

            logits, loss = self._training_step(data, target, grad_step)
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
                    dataloader: DataLoader):
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
            loss = self.cri(logits, labels=target, valid=True)

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

    
