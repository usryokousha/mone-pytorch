import torch
from torchvision.transforms import v2
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
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomResizedCrop(
                img_size,
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
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
                            interpolation=v2.InterpolationMode.BILINEAR,
                        )
                    ],
                    p=0.5,
                )
            )

        # Add normalization transforms
        train_transforms.extend(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

        # Add random erasing
        if random_erase_prob > 0:
            train_transforms.append(v2.RandomErasing(p=random_erase_prob))

        self.train_transforms = v2.Compose(train_transforms)

        # Validation transforms
        self.val_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(
                    img_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                v2.CenterCrop(img_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        """Apply transforms to input image"""
        if self.is_train:
            return self.train_transforms(img)
        return self.val_transforms(img)


class CutMixup:
    def __init__(
        self, cutmix_alpha: float, mixup_alpha: float, num_classes: Optional[int] = None
    ):
        self.cutmix_alpha = cutmix_alpha
        self.mixup_alpha = mixup_alpha
        self.num_classes = num_classes
        self.transforms = v2.RandomChoice(
            [
                v2.CutMix(alpha=self.cutmix_alpha, num_classes=self.num_classes),
                v2.MixUp(alpha=self.mixup_alpha, num_classes=self.num_classes),
            ]
        )

    def __call__(self, img, target):
        return self.transforms(img, target)
