import torch
from torchvision import transforms
from torchvision.transforms import autoaugment, random_erasing
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.v2 import MixUp, CutMix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class ClassificationAugmentation:
    """Augmentation pipeline for image classification using torchvision transforms"""
    
    def __init__(
        self,
        img_size=224,
        randaugment_num_ops=9,
        randaugment_magnitude=0.5,
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        random_erase_prob=0.25,
        label_smoothing=0.1,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.label_smoothing = label_smoothing
        
        # Training transforms
        train_transforms = [
            transforms.RandomResizedCrop(
                img_size, 
                scale=(0.08, 1.0),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
        ]
        
        # Add RandAugment
        if randaugment_num_ops > 0:
            train_transforms.append(
                autoaugment.RandAugment(
                    num_ops=randaugment_num_ops,
                    magnitude=randaugment_magnitude,
                    num_magnitude_bins=31,
                    interpolation=InterpolationMode.BICUBIC
                )
            )
        
        # Add normalization transforms
        train_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        # Add random erasing
        if random_erase_prob > 0:
            train_transforms.append(
                random_erasing.RandomErasing(p=random_erase_prob)
            )
        
        self.train_transforms = transforms.Compose(train_transforms)
        
        # Validation transforms
        self.val_transforms = transforms.Compose([
            transforms.Resize(
                int(img_size * 256/224),  # Resize to 256 if img_size is 224
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # Initialize mix transforms
        mix_transforms = []
        if mixup_alpha > 0:
            mix_transforms.append(MixUp(alpha=mixup_alpha))
        if cutmix_alpha > 0:
            mix_transforms.append(CutMix(alpha=cutmix_alpha))
            
        self.mix_transforms = transforms.Compose(mix_transforms)
        self.has_mix_transforms = len(mix_transforms) > 0

    def __call__(self, img, is_train=True):
        """Apply transforms to input image"""
        if is_train:
            return self.train_transforms(img)
        return self.val_transforms(img)

    def apply_mix_transforms(self, images, labels):
        """Apply mixup/cutmix transforms to batch"""
        if self.has_mix_transforms:
            return self.mix_transforms(images, labels)
        return images, labels


def build_dataloaders(cfg):
    """Build training and validation dataloaders with augmentations"""
    
    augmentation = ClassificationAugmentation(
        img_size=cfg.model.img_size,
        randaugment_num_ops=cfg.augmentation.randaugment.num_ops,
        randaugment_magnitude=cfg.augmentation.randaugment.magnitude,
        randaugment_num_layers=cfg.augmentation.randaugment.num_layers,
        mixup_alpha=cfg.augmentation.mixup.alpha if cfg.augmentation.mixup.enabled else 0.,
        cutmix_alpha=cfg.augmentation.cutmix.alpha if cfg.augmentation.cutmix.enabled else 0.,
        random_erase_prob=cfg.augmentation.random_erase.probability,
        label_smoothing=cfg.augmentation.label_smoothing,
    )

    # Create datasets
    train_dataset = ImageFolder(
        root=cfg.paths.train_dir,
        transform=augmentation
    )
    
    val_dataset = ImageFolder(
        root=cfg.paths.val_dir,
        transform=lambda x: augmentation(x, is_train=False)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, augmentation 