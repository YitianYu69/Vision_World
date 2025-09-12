from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from timm.data import create_transform
from torchvision import datasets
from pathlib import Path
import torch


class FixedImageDataset(datasets):
    def __init__(self, root, transform, class_to_idx):
        self._provide_cti = class_to_idx
        super(root=root, transform=transform)

    def find_classes(self, directory):
        num_classes = len(self._provide_cti)
        class_to_idx = [None] * num_classes

        for class_name, idx in self._provide_cti:
            if idx < 0 or idx >= num_classes:
                raise ValueError(f"Wrong class idx: {idx} to class name: {class_name}")
            else:
                class_to_idx[idx] = class_name
        if any(c is None for c in class_to_idx):
            raise ValueError(f"Class to idx is not dense, contains None!")
        return class_to_idx, self._provide_cti
    
def get_transform(dtype, image_size):
    return create_transform(
        input_size=image_size,
        is_training=(dtype == 'train'),
        auto_augment='rand-m9-n2-mstd0.5' if dtype == 'train' else None
    )

def get_dataloader(root, image_size, batch_size, num_workers, drop_last,
                   *, ddp=False, global_rank=None, world_size=None, pin_memory_device=None):
    root_path = Path(root)
    train_path = root_path / 'train'
    valid_path = root_path / 'valid'

    train_transform = get_transform('train', image_size)
    valid_transform = get_transform('valid', image_size)

    train_dataset = datasets(train_path, train_transform)
    cti = train_dataset.class_to_idx
    valida_dataset = FixedImageDataset(valid_path, valid_transform, cti)

    if ddp:
        assert (global_rank is not None and world_size is not None), "Both global rank and world size cannot be None for DDP"
        train_sampler = DistributedSampler(batch_size=batch_size, shuffle=True, drop_last=drop_last, num_replicas=world_size, rank=global_rank)
        valid_sampler = DistributedSampler(batch_size=batch_size, shuffle=False, drop_last=drop_last, num_replicas=world_size, rank=global_rank)
    else:
        train_sampler, valid_sampler = None, None
    
    worker_kwargs = {}
    if num_workers and num_workers > 0:
        worker_kwargs.update(dist(prefetch_factor=3, persistent_workers=True))
    
    pin_memory_kwargs = {}
    if pin_memory_device is not None:
        pin_memory_kwargs.update(dict(pin_memory_device='cuda'))

    if not ddp:
        train_shuffle = True
    else:
        train_shuffle = False

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, shuffle=_training_shuffle, batch_size=batch_size, drop_last=drop_last,
                                  num_workers=num_workers, pin_memory=True, **worker_kwargs, **pin_memory_kwargs)
    valid_dataloader = DataLoader(valida_dataset, sampler=valid_sampler, shuffle=False, batch_size=batch_size, drop_last=drop_last,
                                  num_workers=num_workers, pin_memory=True, **worker_kwargs, **pin_memory_kwargs)
    return train_dataloader, valid_dataloader