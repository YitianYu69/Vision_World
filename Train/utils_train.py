import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, KLDivLoss

from torch.ao.quantization import get_default_qat, qconfig
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quantize_fx import fuse_fx, prepare_qat_fx

from torchmetrics import Metric, Accuracy, Recall, Precision, F1Score, AUROC

import logging

import math
from contextlib import contextmanager
from typing import List, Union, Callable

logger = logging.getLogger(__name__)

def warmup(model: nn.Module,
           criterion: nn.Module,
           dataset: Dataset, 
           sub_data_portion: float,
           device: str):
    model.train()
    n = range(int(len(dataset) * sub_data_portion))
    data = Subset(dataset, n)
    dl = DataLoader(data, batch_size=32, shuffle=True, drop_last=True, num_workers=2, pin_memory=True, persistent_workers=True)
    model = model.to(device)

    for images, labels in dl:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels=labels)
        loss.backward()

    if torch.cuda.is_available() and str(device).startswith('cuda'):
        torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)

def prefetch(dataloader: DataLoader):
    it = iter(dataloader)

    try:
        return next(it)
    except StopIteration:
        raise RuntimeError("Prefetch failed, dataloader is empty!")

def wrap_model_prepare_qat(model, *, image_size):
    if image_size is None:
        raise ValueError('image_size cannot be None!')

    sample_input = torch.rand((1, 3, image_size, image_size)).to('cuda')

    torch.backends.quantized.engine('fbgemm')

    model.eval()
    qconfig = get_default_qat_qconfig('fbgemm')
    model = fuse_fx(model)
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    model = prepare_qat_fx(model, qconfig_mapping, sample_input)
    return model


def build_CUDA_Graph(model: nn.Module,
                     criterion: nn.Module,
                     dataloader: DataLoader,
                     amp_enable: bool = False,
                     dtype: torch.dtype = torch.float32,
                     device: str = 'cuda',
                     scaler: torch.amp.GradScaler = None,
                     mode: str = 'default',
                     grad_acc_step: int = 1):

    if hasattr(criterion, "graph_mode"):
        criterion.graph_mode = True
                
    device = torch.device(device)
    cpu_x, cpu_y = prefetch(dataloader)
    cuda_x, cuda_y = cpu_x.to(device, non_blocking=True), cpu_y.to(device, non_blocking=True)

    static_x = torch.empty_like(cuda_x)
    static_x.copy_(cuda_x)

    ctype = criterion.criterion_type
    if ctype == 'CE':
        criterion.ensure_buffers_once(grad_acc_step=grad_acc_step, labels=cuda_y)
        criterion.set_batch_target(grad_acc_step=grad_acc_step, labels=cuda_y)
    elif ctype == 'Mixup_Cutmix':
        lam0 = torch.tensor(1.0, dtype=torch.float32, device=device)
        criterion.ensure_buffers_once(grad_acc_step=grad_acc_step, labels=cuda_y, y_a=cuda_y, y_b=cuda_y, lam=lam0)
        criterion.set_batch_target(grad_acc_step=grad_acc_step, labels=cuda_y, y_a=cuda_y, y_b=cuda_y, lam=lam0)
    elif ctype == 'KD':
        with torch.inference_mode():
            out = model(static_x)
        if not isinstance(out, tuple):
            raise RuntimeError('Criterion expects the logits in tuple!')
        _,  logits_kd = out
        if logits_kd.ndim != 2:
            raise RuntimeError(f'KD now only accepts head shape [B, C], got {logits_kd.size()} instead')
        teacher_seed = torch.zeros_like(logits_kd, dtype=criterion.compute_dtype, device=device)
        criterion.ensure_buffers_once(grad_acc_step=grad_acc_step, labels=cuda_y, teacher_logits=teacher_seed)
        criterion.set_batch_target(grad_acc_step=grad_acc_step, labels=cuda_y, teacher_logits=teacher_seed)
    else:
        raise ValueError(f"Unknown criterion type: {ctype}!")
    compute_stream = torch.cuda.Stream(device=device)

    # Build CUDA Graph
    pool = torch.cuda.graph_pool_handle()
    g_no_sync = torch.cuda.CUDAGraph()
    g_sync = torch.cuda.CUDAGraph()
    if amp_enable and dtype == torch.float16:
        logger.info("Warning: with CUDA Graph and float16, please freeze the AMP!")

    model.train()
    model.zero_grad(set_to_none=True)
    with torch.cuda.stream(compute_stream):
        if amp_enable:
            model.require_backward_grad_sync = True
            with torch.cuda.graph(g_sync, pool=pool):
                with torch.amp.autocast(device_type='cuda', dtype=dtype):
                    logits = model(static_x)
                    ori_loss = criterion(logits)
                    loss = ori_loss / criterion.grad_acc_step_buf
                    scaler.scale(loss).backward() if scaler is not None else loss.backward()
            model.require_backward_grad_sync = False
            with torch.cuda.graph(g_no_sync, pool=pool):
                with torch.amp.autocast(device_type='cuda', dtype=dtype):
                    logits = model(static_x)
                    ori_loss = criterion(logits)
        else:
            model.require_backward_grad_sync = True
            with torch.cuda.graph(g_sync, pool=pool):
                logits = model(static_x)
                ori_loss = criterion(logits)
                loss = ori_loss / criterion.grad_acc_step_buf
                loss.backward()
            model.require_backward_grad_sync = False
            with torch.cuda.graph(g_no_sync, pool=pool):
                logits = model(static_x)
                ori_loss = criterion(logits)
    
    return g_sync, g_no_sync, static_x, cuda_y, logits, ori_loss, compute_stream


class setup_criterion(nn.Module):
    def __init__(self, *, labels=None, label_smoothing=None, criterion_type='default', graph_mode=False,
                 y_a=None, y_b=None, lam=None,
                 temperature=None, alpha_kd=0.5, teacher_logits=None, amp=None):
        super().__init__()

        if criterion_type == 'KD':
            assert temperature is not None, 'Variable temperature cannot be None!'
            self.T2 = temperature * temperature
        
        self.alpha_kd = alpha_kd
        self.temperature = temperature
        self.criterion_type = criterion_type

        if amp == "fp16":
            self.compute_dtype = torch.float16
        elif amp == "bf16":
            self.compute_dtype = torch.bfloat16
        else:
            self.compute_dtype = torch.float32

        self.ce = CrossEntropyLoss(label_smoothing=label_smoothing) if label_smoothing else CrossEntropyLoss()
        self.kl = KLDivLoss(reduction='batchmean', log_target=True)

        # Setting up buffers for graph mode:
        self.graph_mode = graph_mode
        if self.graph_mode:
            self.register_buffer('label_buf', None, persistent=False)
            self.register_buffer('y_a_buf', None, persistent=False)
            self.register_buffer('y_b_buf', None, persistent=False)
            self.register_buffer('lam_buf', None, persistent=False)
            self.register_buffer('teacher_logits_buf', None, persistent=False)
            self.register_buffer('grad_acc_step_buf', None, persistent=False)
            logger.info('Graph Mode turned on!')
            
    def _ce(self, logits, *, labels):
        return self.ce(logits, labels)
   
    def _mc(self, logits, *, y_a, y_b, lam):
        logp = F.log_softmax(logits, dim=1)
        loss_a = -logp.gather(1, y_a.view(-1, 1)).squeeze(1)
        loss_b = -logp.gather(1, y_b.view(-1, 1)).squeeze(1)
        return (loss_a.mul_(lam).add_(loss_b, alpha=(1 - lam))).mean()

    def _kd(self, logits, *, labels, teacher_logits):
        assert isinstance(logits, tuple), 'Please use a tuple of logits'
        logits, logits_kd = logits
        log_p_s = F.log_softmax(logits_kd.float() / self.temperature, dim=1)
        q_s = F.log_softmax(teacher_logits.float() / self.temperature, dim=1)
        soft_loss = self.kl(log_p_s, q_s) * self.T2
        hard_loss = self.ce(logits, labels)
        return soft_loss.mul_(self.alpha_kd).add_(hard_loss, alpha=(1 - self.alpha_kd))

    @torch.no_grad()
    def ensure_buffers_once(self, *, grad_acc_step, labels, y_a=None, y_b=None, lam=None, teacher_logits=None):
        self.grad_acc_step_buf = torch.empty_like(torch.tensor(1.0, dtype=torch.float32, device=labels.device))
        if self.criterion_type == 'CE' and labels is not None:
            self.label_buf = torch.empty_like(labels, dtype=torch.long, device=labels.device)
        elif self.criterion_type == 'Mixup_Cutmix' and None not in [y_a, y_b, lam]:
            self.y_a_buf = torch.empty_like(y_a, dtype=torch.long, device=y_a.device)
            self.y_b_buf = torch.empty_like(y_b, dtype=torch.long, device=y_b.device)
            self.lam_buf = torch.empty_like(lam, dtype=torch.float32, device=lam.device)
        elif self.criterion_type == 'KD' and labels is not None and teacher_logits is not None:
            self.teacher_logits_buf = torch.empty_like(teacher_logits, dtype=self.compute_dtype, device=teacher_logits.device)
            self.label_buf = torch.empty_like(labels, dtype=torch.long, device=labels.device)

    @torch.no_grad()
    def set_batch_target(self, *, grad_acc_step, labels=None, y_a=None, y_b=None, lam=None, teacher_logits=None):
        self.grad_acc_step_buf.copy_(grad_acc_step)
        if labels is not None: self.label_buf.copy_(labels)
        if None not in [y_a, y_b, lam]:
            self.y_a_buf.copy_(y_a)
            self.y_b_buf.copy_(y_b)
            self.lam_buf.copy_(lam)
        if teacher_logits is not None: self.teacher_logits_buf.copy_(teacher_logits)

    def forward(self, logits, *, labels=None, y_a=None, y_b=None, lam=None, teacher_logits=None, valid=False):
        if self.graph_mode:
            if self.criterion_type == 'CE' or valid:
                return self._ce(logits, labels=self.label_buf)
            elif self.criterion_type == 'Mixup_Cutmix':
                return self._mc(logits, y_a=self.y_a_buf, y_b=self.y_b_buf, lam=self.lam_buf)
            elif self.criterion_type == 'KD':
                return self._kd(logits, labels=self.label_buf, teacher_logits=self.teacher_logits_buf)
        else:
            if self.criterion_type == 'CE' or valid:
                return self._ce(logits, labels=labels)
            elif self.criterion_type == 'Mixup_Cutmix':
                return self._mc(logits, y_a=y_a, y_b=y_b, lam=lam)
            elif self.criterion_type == 'KD':
                return self._kd(logits, labels=labels, teacher_logits=teacher_logits)
            

    

def build_metrics(*, metric_lists: List[str],
                  task:str,
                  num_classes: int,
                  average_type: str,
                  sync: bool,
                  device: Union[str, torch.device] = "cpu"):
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
                 device = None,
                 use_fp32_master = True,
                 include_buffers = True,
                 param_filters = None,
                 update_every = 1,
                 kahan_compensation: bool = False):

        self.ema_model = self._unwrap_model(model)
        self.decay_base = float(decay)
        self.tau = int(tau)
        self.device = device
        self.use_fp32_master = use_fp32_master
        self.include_buffers = include_buffers
        self.param_filters = param_filters or (lambda n, p: p.requires_grad and p.is_floating_point())
        self.update_every = update_every
        self.num_updates = 0

        self._decay_prod = 1.0

        self.shadow_params = {}
        self.kahan_comp = {} if kahan_compensation else None
        self.shadow_buffers = {}

        self.backed_params = {}
        self.backed_buffers = {}

        self._register(self.ema_model)

    def _unwrap_model(self, model):
        return model.module if hasattr(model,'module') else model
    
    def _register(self, model):
        self.shadow_params.clear()
        self.shadow_buffers.clear()

        if self.kahan_comp is not None:
            self.kahan_comp.clear()

        for n, p in self._named_fparameters(model):
            ema_p = p.detach().clone()
            if self.use_fp32_master:
                ema_p = ema_p.to(torch.float32)
            if self.device is not None:
                ema_p = ema_p.to(self.device)
            ema_p.requires_grad_(False)
            self.shadow_params[n] = ema_p
            # For Kahan
            if self.kahan_comp is not None:
                self.kahan_comp[n] = torch.zeros(ema_p.size(), dtype=ema_p.dtype, device=ema_p.device)

        if self.include_buffers:
            for n, b in model.named_buffers():
                ema_b = b.detach().clone()
                if self.device is not None:
                    ema_b = ema_b.to(self.device)
                self.shadow_buffers[n] = ema_b
    
    def to(self, device):
        self.device = device
        for n in self.shadow_params.keys():
            self.shadow_params[n] = self.shadow_params[n].to(self.device)
            if self.kahan_comp is not None:
                self.kahan_comp[n] = self.kahan_comp[n].to(self.device)

        if self.include_buffers:
            for n in self.shadow_buffers.keys():
                self.shadow_buffers[n] = self.shadow_buffers[n].to(self.device)
    
    def _is_bn_stats(self, name):
        return name.endswith('running_mean') or name.endswith('running_var')

    @torch.no_grad()
    def update_parameters(self, model):
        if (self.num_updates + 1) % self.update_every != 0:
            return
        

        model = self._unwrap_model(model)
        c_decay = self._current_decay()

        for n, p, ema_p in self._iter_named_fparameters_with_ema(model):
            if p is None or ema_p is None:
                continue

            # Ensure devices match; if user forced a different device via .to(),
            # this will raise early instead of silently syncing each step.
            if p.device != ema_p.device:
                raise RuntimeError(
                    f"EMA param {n} on {ema_p.device} but model param on {p.device}. "
                    "Keep EMA on the same device for fast implicit promotion."
                )
            
            if self.kahan_comp is not None:
                self.kahan_add_(ema_p, self.kahan_comp[n], p.detach(), c_decay)
            else:
                ema_p.mul_(c_decay).add_(p.detach(), alpha=1 - c_decay)

        if self.include_buffers:
            for n, b, ema_b in self._iter_named_buffers_with_ema(model):
                if b is None or ema_b is None:
                    continue
                    
                if b.device != ema_b.device:
                    raise RuntimeError(
                        f"EMA buffer {n} on {ema_b.device} but model buffer on {b.device}"
                    )
                
                if self._is_bn_stats(n):
                    ema_b.mul_(c_decay).add_(b.detach(), alpha=1.0 - c_decay)
                else:
                    ema_b.copy_(b.detach())

        
        self.num_updates += 1
        self._decay_prod *= c_decay
    
    @torch.no_grad()
    def kahan_add_(self, ema_p, kahan_comp, p, decay):
        y = (1.0 - decay) * p - kahan_comp
        t = ema_p.mul_(decay) + y
        kahan_comp.copy_((t - ema_p) - y)
        ema_p.copy_(t)

    def _bias_correction(self):
        # For the time-varying decay 
        return 1.0 - (self._decay_prod) if self.num_updates > 0 else 1.0

    # Overwrite model's weights with ema_model's weights for evaluation
    @torch.no_grad()
    def copy_to(self, model, *, bias_correction=False):
        model = self._unwrap_model(model)

        corr = 1.0
        if bias_correction:
            corr = 1.0 / max(self._bias_correction(), 1e-8)
        for n, p, ema_p in self._iter_named_fparameters_with_ema(model):
            if ema_p is not None:
                p.copy_((ema_p * corr).to(device=p.device, dtype=p.dtype))
            
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

    @contextmanager
    def average_parameters(self, model, *, bias_correction=False):
        self.store(model)
        try:
            self.copy_to(model, bias_correction=bias_correction)
            yield
        finally:
            self.restore(model)


