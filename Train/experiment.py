import torch
import math
from contextlib import contextmanager


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
                 kahan_compensation: bool = False,
                 foreach: bool = False):

        self.ema_model = self._unwrap_model(model)
        self.decay_base = float(decay)
        self.tau = int(tau)
        self.device = device
        self.use_fp32_master = use_fp32_master
        self.include_buffers = include_buffers
        self.param_filters = param_filters or (lambda n, p: p.requires_grad and p.is_floating_point())
        self.update_every = update_every
        self.num_updates = 0
        self.foreach = foreach

        self._decay_prod = 1.0

        self.shadow_params = None
        self.shadow_p_view = None
        self.kahan_comp = [] if kahan_compensation else None
        self.shadow_buffers = {}

        self.backed_params = {}
        self.backed_buffers = {}

        if kahan_compensation and foreach:
            raise ValueError("Kahan Summation only supports flat contiguous updates; set foreach=False.")

        self._register(self.ema_model)

    def _unwrap_model(self, model):
        return model.module if hasattr(model,'module') else model
    
    def _register(self, model):
        self.shadow_params = None
        self.shadow_p_view = None
        self.shadow_buffers.clear()
        want_kahan = self.kahan_comp is not None
        self.kahan_comp = None

        # Create a temp ema p for later process
        temp_params, params = [], []
        for n, p in self._named_fparameters(model):
            ema_p = p.detach()
            if self.use_fp32_master:
                ema_p = ema_p.to(torch.float32)
            ema_p.requires_grad_(False)
            if self.device is not None:
                ema_p = ema_p.to(device=self.device)
            temp_params.append(ema_p)
            params.append(p)


        self.shadow_params, self.shadow_p_view = self._pack_flat(temp_params, params)
        # For Kahan
        if want_kahan:
            self.kahan_comp = torch.zeros_like(self.shadow_params)

        if self.include_buffers:
            for n, b in model.named_buffers():
                ema_b = b.detach().clone()
                if self.device is not None:
                    ema_b = ema_b.to(self.device)
                self.shadow_buffers[n] = ema_b


    @torch.no_grad()
    def _pack_flat(self, tensors, ref_tensor=None):
        if  tensors is None or len(tensors) == 0:
            raise RuntimeError("Please check your model: _pack_flat expects a non-empty array of tensors. This is not the code's problem!")
        if ref_tensor is not None and len(ref_tensor) != len(tensors):
            raise RuntimeError("EMA._pack_flat: ref_tensor and tensors length mismatch.")

        flats = [t.contiguous().view(-1) for t in tensors]
        flat = torch.empty(sum(x.numel() for x in flats), dtype=flats[0].dtype, device=flats[0].device)
        offset = 0
        view = []

        _iter = zip(ref_tensor, flats) if ref_tensor is not None else ((t,) for t in flats)
        for *r_t, t in _iter:
            count = t.numel()
            flat[offset : offset + count].copy_(t)
            if ref_tensor is not None:
                (r_t,) = r_t
                view.append(flat[offset : offset + count].view_as(r_t))
            offset += count

        if ref_tensor is not None:
            return flat, view
        else:
            return flat
    
    def to(self, device, *, model=None):
        if model is None:
            raise ValueError(f"Moving EMA requires rebuilding the views; pass the model to rebuild them; Reduce using it, it's expensive!")

        self.device = device
        if self.shadow_params is not None:
            offset, view = 0, []
            model = self._unwrap_model(model)
            self.shadow_params = self.shadow_params.to(self.device)
            for n, p in self._named_fparameters(model):
                count = p.numel()
                view.append(self.shadow_params[offset : offset + count].view_as(p))
                offset += count
            if offset != self.shadow_params.numel():
                raise RuntimeError("EMA.to(): model param size changed; re-register EMA before moving devices.")
            self.shadow_p_view = view
        if self.kahan_comp is not None:
            self.kahan_comp = self.kahan_comp.to(self.device)

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

        if self.foreach:
            params = [p.detach() if not self.use_fp32_master else p.detach().float() for n, p in self._named_fparameters(model)]
            if not params:
                raise RuntimeError("Current params for update EMA is empty, please double check your model!")
            if len(params) != len(self.shadow_p_view):  # <--- add this
                raise RuntimeError("EMA: parameter count changed; re-register EMA with the current model.")
            if params[0].device != self.shadow_params.device:
                raise RuntimeError(f'EMA params are on {self.shadow_params.device}, but model params are on {params[0].device}.')
            if not self.use_fp32_master and params[0].dtype != self.shadow_params.dtype:
                raise RuntimeError(f'EMA params are on {self.shadow_params.dtype}, but model params are on {params[0].dtype}.')
        else:
            params = self._pack_flat([p.detach() for n, p in self._named_fparameters(model)])
            if params.numel() != self.shadow_params.numel():
                raise RuntimeError("EMA: total parameter size changed; re-register EMA with the current model.")
            if params.device != self.shadow_params.device:
                raise RuntimeError(f"EMA params are on {self.shadow_params.device} but model params are on {params.device}.")
            if params.dtype != self.shadow_params.dtype:
                if self.use_fp32_master:
                    params = params.to(dtype=self.shadow_params.dtype)
                else:
                    raise RuntimeError("Flat EMA update requires dtype match; set use_fp32_master=True or ensure matching dtype.")
            
        if self.kahan_comp is not None:
            self.kahan_add_(self.shadow_params, self.kahan_comp, params, c_decay)
        else:
            if self.foreach:
                torch._foreach_mul_(self.shadow_p_view, c_decay)
                torch._foreach_add_(self.shadow_p_view, params, alpha=1.0 - c_decay)
            else:
                self.shadow_params.mul_(c_decay).add_(params, alpha=1.0 - c_decay)

        if self.include_buffers:
            for n, b, ema_b in self._iter_named_buffers_with_ema(model):
                if b is None or ema_b is None:
                    continue
                    
                if b.device != ema_b.device:
                    raise RuntimeError(
                        f"EMA buffer {n} on {ema_b.device} but model buffer on {b.device}"
                    )
                if self._is_bn_stats(n):
                    ema_b.mul_(c_decay).add_(b.detach(), alpha=1 - c_decay)
                else:
                    ema_b.copy_(b.detach())
        
        self.num_updates += 1
        self._decay_prod *= c_decay
    
    @torch.no_grad()
    def kahan_add_(self, ema_p, kahan_comp, p, decay):
        y = p * (1.0 - decay) - kahan_comp
        t = ema_p.mul_(decay) + y
        kahan_comp.copy_((t - ema_p) - y)
        ema_p.copy_(t)

    def _bias_correction(self):
        # For the time-varying decay 
        return 1.0 - (self._decay_prod) if self.num_updates > 0 else 1.0

    # Overwrite model's weights with ema_model's weights for evaluation
    # bias_correction=True is intended only if shadow_params were zero-initialized
    @torch.no_grad()
    def copy_to(self, model, *, bias_correction=False):
        model = self._unwrap_model(model)

        # O(1) count check to avoid silent zip truncation
        n_params = sum(1 for _ in self._named_fparameters(model))
        if n_params != len(self.shadow_p_view):
            raise RuntimeError("EMA.copy_to(): parameter count changed; re-register EMA.")

        corr = 1.0
        if bias_correction:
            corr = 1.0 / max(self._bias_correction(), 1e-8)
        for n, p, ema_view in self._iter_named_fparameters_with_ema_view(model):
            p.copy_((ema_view * corr).to(device=p.device, dtype=p.dtype))
            
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

    def _iter_named_fparameters_with_ema_view(self, model):
        for (n, p), ema_view in zip(self._named_fparameters(model), self.shadow_p_view ):
            yield(n, p, ema_view)

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