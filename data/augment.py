# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
from copy import deepcopy
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)
DEFAULT_CROP_FRACTION = 1.0




class Compose:
    """Class for composing multiple image transformations."""

    def __init__(self, transforms):
        """Initializes the Compose object with a list of transforms."""
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """Applies a series of transformations to input data."""
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """Appends a new transform to the existing list of transforms."""
        self.transforms.append(transform)

    def insert(self, index, transform):
        """Inserts a new transform to the existing list of transforms."""
        self.transforms.insert(index, transform)

    def __getitem__(self, index: Union[list, int]) -> "Compose":
        """Retrieve a specific transform or a set of transforms using indexing."""
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        index = [index] if isinstance(index, int) else index
        return Compose([self.transforms[i] for i in index])

    def __setitem__(self, index: Union[list, int], value: Union[list, int]) -> None:
        """Retrieve a specific transform or a set of transforms using indexing."""
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        if isinstance(index, list):
            assert isinstance(
                value, list
            ), f"The indices should be the same type as values, but got {type(index)} and {type(value)}"
        if isinstance(index, int):
            index, value = [index], [value]
        for i, v in zip(index, value):
            assert i < len(self.transforms), f"list index {i} out of range {len(self.transforms)}."
            self.transforms[i] = v

    def tolist(self):
        """Converts the list of transforms to a standard Python list."""
        return self.transforms

    def __repr__(self):
        """Returns a string representation of the object."""
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"



# Classification augmentations -----------------------------------------------------------------------------------------
def classify_transforms(
    size=224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    interpolation=Image.BILINEAR,
    crop_fraction: float = DEFAULT_CROP_FRACTION,
):
    """
    Classification transforms for evaluation/inference. Inspired by timm/data/transforms_factory.py.

    Args:
        size (int): image size
        mean (tuple): mean values of RGB channels
        std (tuple): std values of RGB channels
        interpolation (T.InterpolationMode): interpolation mode. default is T.InterpolationMode.BILINEAR.
        crop_fraction (float): fraction of image to crop. default is 1.0.

    Returns:
        (T.Compose): torchvision transforms
    """
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    if isinstance(size, (tuple, list)):
        assert len(size) == 2
        scale_size = tuple(math.floor(x / crop_fraction) for x in size)
    else:
        scale_size = math.floor(size / crop_fraction)
        scale_size = (scale_size, scale_size)

    # Aspect ratio is preserved, crops center within image, no borders are added, image is lost
    if scale_size[0] == scale_size[1]:
        # Simple case, use torchvision built-in Resize with the shortest edge mode (scalar size arg)
        tfl = [T.Resize(scale_size[0], interpolation=interpolation)]
    else:
        # Resize the shortest edge to matching target dim for non-square target
        tfl = [T.Resize(scale_size)]
    tfl += [T.CenterCrop(size)]

    tfl += [
        T.ToTensor(),
        T.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std),
        ),
    ]

    return T.Compose(tfl)


# Classification training augmentations --------------------------------------------------------------------------------
def classify_augmentations(
    size=224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    auto_augment=None,
    hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
    hsv_s=0.4,  # image HSV-Saturation augmentation (fraction)
    hsv_v=0.4,  # image HSV-Value augmentation (fraction)
    force_color_jitter=False,
    erasing=0.0,
    interpolation=Image.BILINEAR,
):
    """
    Classification transforms with augmentation for training. Inspired by timm/data/transforms_factory.py.

    Args:
        size (int): image size
        scale (tuple): scale range of the image. default is (0.08, 1.0)
        ratio (tuple): aspect ratio range of the image. default is (3./4., 4./3.)
        mean (tuple): mean values of RGB channels
        std (tuple): std values of RGB channels
        hflip (float): probability of horizontal flip
        vflip (float): probability of vertical flip
        auto_augment (str): auto augmentation policy. can be 'randaugment', 'augmix', 'autoaugment' or None.
        hsv_h (float): image HSV-Hue augmentation (fraction)
        hsv_s (float): image HSV-Saturation augmentation (fraction)
        hsv_v (float): image HSV-Value augmentation (fraction)
        force_color_jitter (bool): force to apply color jitter even if auto augment is enabled
        erasing (float): probability of random erasing
        interpolation (T.InterpolationMode): interpolation mode. default is T.InterpolationMode.BILINEAR.

    Returns:
        (T.Compose): torchvision transforms
    """
    # Transforms to apply if Albumentations not installed
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    if not isinstance(size, int):
        raise TypeError(f"classify_transforms() size {size} must be integer, not (list, tuple)")
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
    primary_tfl = [T.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)]
    if hflip > 0.0:
        primary_tfl += [T.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [T.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str)
        # color jitter is typically disabled if AA/RA on,
        # this allows override without breaking old hparm cfgs
        disable_color_jitter = not force_color_jitter

        if auto_augment == "randaugment":
            if TORCHVISION_0_11:
                secondary_tfl += [T.RandAugment(interpolation=interpolation)]
            else:
                LOGGER.warning('"auto_augment=randaugment" requires torchvision >= 0.11.0. Disabling it.')

        elif auto_augment == "augmix":
            if TORCHVISION_0_13:
                secondary_tfl += [T.AugMix(interpolation=interpolation)]
            else:
                LOGGER.warning('"auto_augment=augmix" requires torchvision >= 0.13.0. Disabling it.')

        elif auto_augment == "autoaugment":
            if TORCHVISION_0_10:
                secondary_tfl += [T.AutoAugment(interpolation=interpolation)]
            else:
                LOGGER.warning('"auto_augment=autoaugment" requires torchvision >= 0.10.0. Disabling it.')

        else:
            raise ValueError(
                f'Invalid auto_augment policy: {auto_augment}. Should be one of "randaugment", '
                f'"augmix", "autoaugment" or None'
            )

    if not disable_color_jitter:
        secondary_tfl += [T.ColorJitter(brightness=hsv_v, contrast=hsv_v, saturation=hsv_s, hue=hsv_h)]

    final_tfl = [
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        T.RandomErasing(p=erasing, inplace=True),
    ]

    return T.Compose(primary_tfl + secondary_tfl + final_tfl)


# NOTE: keep this class for backward compatibility
class ClassifyLetterBox:
    """
    YOLOv8 LetterBox class for image preprocessing, designed to be part of a transformation pipeline, e.g.,
    T.Compose([LetterBox(size), ToTensor()]).

    Attributes:
        h (int): Target height of the image.
        w (int): Target width of the image.
        auto (bool): If True, automatically solves for short side using stride.
        stride (int): The stride value, used when 'auto' is True.
    """

    def __init__(self, size=(640, 640), auto=False, stride=32):
        """
        Initializes the ClassifyLetterBox class with a target size, auto-flag, and stride.

        Args:
            size (Union[int, Tuple[int, int]]): The target dimensions (height, width) for the letterbox.
            auto (bool): If True, automatically calculates the short side based on stride.
            stride (int): The stride value, used when 'auto' is True.
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):
        """
        Resizes the image and pads it with a letterbox method.

        Args:
            im (numpy.ndarray): The input image as a numpy array of shape HWC.

        Returns:
            (numpy.ndarray): The letterboxed and resized image as a numpy array.
        """
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old dimensions
        h, w = round(imh * r), round(imw * r)  # resized image dimensions

        # Calculate padding dimensions
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else (self.h, self.w)
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)

        # Create padded image
        im_out = np.full((hs, ws, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


# NOTE: keep this class for backward compatibility
class CenterCrop:
    """YOLOv8 CenterCrop class for image preprocessing, designed to be part of a transformation pipeline, e.g.,
    T.Compose([CenterCrop(size), ToTensor()]).
    """

    def __init__(self, size=640):
        """Converts an image from numpy array to PyTorch tensor."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):
        """
        Resizes and crops the center of the image using a letterbox method.

        Args:
            im (numpy.ndarray): The input image as a numpy array of shape HWC.

        Returns:
            (numpy.ndarray): The center-cropped and resized image as a numpy array.
        """
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


# NOTE: keep this class for backward compatibility
class ToTensor:
    """YOLOv8 ToTensor class for image preprocessing, i.e., T.Compose([LetterBox(size), ToTensor()])."""

    def __init__(self, half=False):
        """Initialize YOLOv8 ToTensor object with optional half-precision support."""
        super().__init__()
        self.half = half

    def __call__(self, im):
        """
        Transforms an image from a numpy array to a PyTorch tensor, applying optional half-precision and normalization.

        Args:
            im (numpy.ndarray): Input image as a numpy array with shape (H, W, C) in BGR order.

        Returns:
            (torch.Tensor): The transformed image as a PyTorch tensor in float32 or float16, normalized to [0, 1].
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
