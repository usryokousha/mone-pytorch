from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, ImageNet, CIFAR100

from mone_pytorch.utils import augmentation

def build_dataloaders(cfg):
    # Create augmentation instance
    train_augmentation = augmentation.ClassificationAugmentation(
        img_size=cfg.model.img_size,
        randaugment_num_ops=cfg.augmentation.randaugment.num_ops,
        randaugment_magnitude=cfg.augmentation.randaugment.magnitude,
        random_erase_prob=cfg.augmentation.random_erase.probability,
        is_train=True,
        mean=augmentation.mean_std[cfg.data.dataset_name][0],
        std=augmentation.mean_std[cfg.data.dataset_name][1]
    )
    val_augmentation = augmentation.ClassificationAugmentation(
        img_size=cfg.model.img_size,
        is_train=False,
        mean=augmentation.mean_std[cfg.data.dataset_name][0],
        std=augmentation.mean_std[cfg.data.dataset_name][1]
    )

    if 'imagenet1k' == cfg.data.dataset_name:
        # Create datasets
        train_dataset = ImageNet(
            root=cfg.data.path,
            split="train",
            transform=train_augmentation
        )
    
        val_dataset = ImageNet(
            root=cfg.data.path,
            split="val",
            transform=val_augmentation
        )
    elif 'cifar100' == cfg.data.dataset_name:
        train_dataset = CIFAR100(
            root=cfg.data.path,
            train=True,
            transform=train_augmentation
        )
        val_dataset = CIFAR100(
            root=cfg.data.path,
            train=False,
            transform=val_augmentation
        )
    elif 'imagenet21k' == cfg.data.dataset_name:
        train_dataset = ImageFolder(
            root=cfg.data.train_path,
            transform=train_augmentation
        )
        val_dataset = ImageFolder(
            root=cfg.data.val_path,
            transform=val_augmentation
        )
    else:
        raise ValueError(f"Config file does not contain valid dataset")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader