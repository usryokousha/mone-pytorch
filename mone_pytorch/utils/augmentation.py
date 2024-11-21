from torchvision import transforms
from torchvision.transforms import autoaugment, RandomErasing
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from typing import Optional

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ClassificationAugmentation:
    """Augmentation pipeline for image classification using torchvision transforms"""

    def __init__(
        self,
        img_size=224,
        randaugment_num_ops=2,
        randaugment_magnitude=9,
        random_erase_prob=0.0,
        is_train=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):
        self.is_train = is_train
        # Training transforms
        train_transforms = [
            v2.RandomResizedCrop(
                img_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            v2.RandomHorizontalFlip(),
        ]

        # Add RandAugment
        if randaugment_num_ops > 0:
            train_transforms.append(
                v2.RandomApply(
                    [
                        v2.RandAugment(
                            num_ops=randaugment_num_ops,
                            magnitude=randaugment_magnitude,
                            num_magnitude_bins=31,
                            interpolation=InterpolationMode.BICUBIC,
                        )
                    ],
                    p=0.5,
                )
            )

        # Add normalization transforms
        train_transforms.extend(
            [
                v2.ToTensor(),
                v2.Normalize(mean=mean, std=std),
            ]
        )

        # Add random erasing
        if random_erase_prob > 0:
            train_transforms.append(RandomErasing(p=random_erase_prob))

        self.train_transforms = v2.Compose(train_transforms)

        # Validation transforms
        self.val_transforms = v2.Compose(
            [
                v2.Resize(
                    int(img_size * 256 / 224),  # Resize to 256 if img_size is 224
                    interpolation=InterpolationMode.BICUBIC,
                ),
                v2.CenterCrop(img_size),
                v2.ToTensor(),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        """Apply transforms to input image"""
        if self.is_train:
            return self.train_transforms(img)
        return self.val_transforms(img)


class CutMixup:
    def __init__(self, alpha: float, num_classes: Optional[int] = None):
        self.alpha = alpha
        self.num_classes = num_classes
        self.transforms = v2.RandomChoice([v2.CutMix(alpha=alpha, num_classes=num_classes), 
                                           v2.MixUp(alpha=alpha, num_classes=num_classes)])

    def __call__(self, img, target):
        return self.transforms(img, target)


def build_dataloaders(cfg):
    """Build training and validation dataloaders with augmentations"""

    train_augmentation = ClassificationAugmentation(
        img_size=cfg.model.img_size,
        randaugment_num_ops=cfg.augmentation.randaugment.num_ops,
        randaugment_magnitude=cfg.augmentation.randaugment.magnitude,
        random_erase_prob=cfg.augmentation.random_erase.probability,
        is_train=True,
    )

    val_augmentation = ClassificationAugmentation(
        img_size=cfg.model.img_size, is_train=False
    )

    # Create datasets
    train_dataset = ImageFolder(root=cfg.paths.train_dir, transform=train_augmentation)
    val_dataset = ImageFolder(root=cfg.paths.val_dir, transform=val_augmentation)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
