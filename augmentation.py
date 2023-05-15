# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageOps, ImageEnhance

class Resize(object):
    def __init__(self, scale=(512,512),keep_ratio=False):
        self.scale = scale
        self.keep_ratio = keep_ratio
    def __call__(self, result):
        img = result['img']
        label = result['label']

        new_w,new_h = self.scale
        w,h = img.size
        if label:
            assert img.size == label.size
        if self.keep_ratio:
            max_long_edge = max(self.scale)
            max_short_edge = min(self.scale)
            scale_factor = min(max_long_edge / max(h, w),max_short_edge / min(h, w))
            new_w,new_h = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
        result['img'] = img.resize((new_w,new_h), Image.BILINEAR)
        if label:
            result['label'] = label.resize((new_w,new_h), Image.NEAREST)

        return result
    
class ResizeStepScaling(object):
    """
    Scale an image proportionally within a range.

    Args:
        min_scale_factor (float, optional): The minimum scale. Default: 0.75.
        max_scale_factor (float, optional): The maximum scale. Default: 1.25.
        scale_step_size (float, optional): The scale interval. Default: 0.25.

    Raises:
        ValueError: When min_scale_factor is smaller than max_scale_factor.
    """

    def __init__(self,
                 min_scale_factor=0.75,
                 max_scale_factor=1.25,
                 scale_step_size=0.25):
        if min_scale_factor > max_scale_factor:
            raise ValueError(
                'min_scale_factor must be less than max_scale_factor, '
                'but they are {} and {}.'.format(min_scale_factor,
                                                 max_scale_factor))
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_step_size = scale_step_size

    def __call__(self, result):
        img = result['img']
        label = result['label']
        w, h = img.size 
        if label:
            assert img.size == label.size, "img size equal to label size."
        if self.min_scale_factor == self.max_scale_factor:
            scale_factor = self.min_scale_factor

        elif self.scale_step_size == 0:
            scale_factor = np.random.uniform(self.min_scale_factor,
                                             self.max_scale_factor)

        else:
            num_steps = int((self.max_scale_factor - self.min_scale_factor) /
                            self.scale_step_size + 1)
            scale_factors = np.linspace(self.min_scale_factor,
                                        self.max_scale_factor,
                                        num_steps).tolist()
            np.random.shuffle(scale_factors)
            scale_factor = scale_factors[0]
        w = int(round(scale_factor * w))
        h = int(round(scale_factor * h))

        result['img'] = img.resize((w, h), Image.BILINEAR)
        if label:
            result['label'] = label.resize((w,h), Image.NEAREST)
        return result
    
class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, result):
        img = result['img']
        label = result['label']
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            if label:
                label = ImageOps.expand(label, border=self.padding, fill=0)
        if label:
            assert img.size == label.size
        w, h = img.size
        tw, th = self.size
        if w == tw and h == th:
            return result
        if w < tw or h < th:
            result['img'] = result['img'].resize((tw, th), Image.BILINEAR)
            if result['label']:
                result['label'] = result['label'].resize((tw, th), Image.NEAREST)
            return result

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        result['img'] = img.crop((x1, y1, x1 + tw, y1 + th))
        if label: 
            result['label'] = label.crop((x1, y1, x1 + tw, y1 + th))
        return result
    
class RandomPaddingCrop:
    def __init__(self,
                 crop_size=(512, 512),
                 im_padding_value=128,
                 label_padding_value=255,
                 ignore_index=255):
        self.crop_size = crop_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value
        self.ignore_index = ignore_index

    def __call__(self, result):
        w, h = result['img'].size 
        pad_height = max(self.crop_size[0] - h, 0)
        pad_width = max(self.crop_size[1] - w, 0) 
        if (pad_height > 0 or pad_width > 0):
            result['img'] = ImageOps.expand(result['img'], border=(0,0,pad_width,pad_height), fill=self.im_padding_value)
            if result['label']:
                result['label'] = ImageOps.expand(result['label'], border=(0,0,pad_width,pad_height), fill=self.label_padding_value)
        
        w, h = result['img'].size
        if w == self.crop_size[0] and h == self.crop_size[1]:
            return result
        
        x1 = random.randint(0, w - self.crop_size[0])
        y1 = random.randint(0, h - self.crop_size[1])
        result['img'] = result['img'].crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        if result['label']: 
            result['label'] = result['label'].crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        return result

class RandomDistort:
    """
    Distort an image with random configurations.

    Args:
        brightness_range (float, optional): A range of brightness. Default: 0.5.
        brightness_prob (float, optional): A probability of adjusting brightness. Default: 0.5.
        contrast_range (float, optional): A range of contrast. Default: 0.5.
        contrast_prob (float, optional): A probability of adjusting contrast. Default: 0.5.
        saturation_range (float, optional): A range of saturation. Default: 0.5.
        saturation_prob (float, optional): A probability of adjusting saturation. Default: 0.5.
        hue_range (int, optional): A range of hue. Default: 18.
        hue_prob (float, optional): A probability of adjusting hue. Default: 0.5.
        sharpness_range (float, optional): A range of sharpness. Default: 0.5.
        sharpness_prob (float, optional): A probability of adjusting saturation. Default: 0.
    """

    def __init__(self,
                 brightness_range=0.5,
                 brightness_prob=0.5,
                 contrast_range=0.5,
                 contrast_prob=0.5,
                 saturation_range=0.5,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5,
                 sharpness_range=0.5,
                 sharpness_prob=0):
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.saturation_range = saturation_range
        self.saturation_prob = saturation_prob
        self.hue_range = hue_range
        self.hue_prob = hue_prob
        self.sharpness_range = sharpness_range
        self.sharpness_prob = sharpness_prob

    def _brightness(self,im,brightness_range):
        brightness_delta = np.random.uniform(1-brightness_range, 1+brightness_range)
        im = ImageEnhance.Brightness(im).enhance(brightness_delta)
        return im
    def _contrast(self,im, contrast_range):
        contrast_delta = np.random.uniform(1-contrast_range,1+contrast_range)
        im = ImageEnhance.Contrast(im).enhance(contrast_delta)
        return im
    def _saturation(self,im, saturation_range):
        saturation_delta = np.random.uniform(1-saturation_range,1+saturation_range)
        im = ImageEnhance.Color(im).enhance(saturation_delta)
        return im
    def _sharpness(self,im, sharpness_range):
        sharpness_delta = np.random.uniform(1-sharpness_range,1+sharpness_range)
        im = ImageEnhance.Sharpness(im).enhance(sharpness_delta)
        return im
    def _hue(self,im, hue_range):
        hue_delta = np.random.uniform(-hue_range,hue_range)
        im = np.array(im.convert('HSV'))
        im[:, :, 0] = im[:, :, 0] + hue_delta
        im = Image.fromarray(im, mode='HSV').convert('RGB')
        return im
    def __call__(self,result):
        img = result['img']
        ops = [self._brightness,self._contrast,self._saturation,self._sharpness,self._hue]
        random.shuffle(ops)
        prob_dict = {
            'brightness': self.brightness_prob,
            'contrast': self.contrast_prob,
            'saturation': self.saturation_prob,
            'hue': self.hue_prob,
            'sharpness': self.sharpness_prob
        }
        param_dict = {
            'brightness': self.brightness_range,
            'contrast': self.contrast_range,
            'saturation': self.saturation_range,
            'hue': self.hue_range,
            'sharpness': self.sharpness_range
        }
        for i in range(len(ops)):
            prob = prob_dict[ops[i].__name__[1:]]
            if np.random.uniform(0, 1) < prob:
                param = param_dict[ops[i].__name__[1:]]
                img = ops[i](img,param)
        result['img'] = img
        return result        

class RandomFlip(object):   
    def __call__(self, result):
        if random.random() < 0.5:
            result['img'] = result['img'].transpose(Image.FLIP_LEFT_RIGHT)
            if result['label']:
                result['label'] = result['label'].transpose(Image.FLIP_LEFT_RIGHT) #left or right
            return result
        return result
    

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0,1.0,1.0)):
        self.mean = mean
        self.std = std

    def __call__(self, result):
        if min(result['label'].shape)!=0:
            result['label'] = result['label'].long()
        result['img'] = F.normalize(result['img'], self.mean, self.std).float()

        return result


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, result):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        result['img'] = torch.from_numpy(np.array(result['img'],np.float32).transpose((2, 0, 1))).div(255)
        
        if result['label']:
            result['label'] = torch.from_numpy(np.array(result['label'],np.int64))

        return result

class RandomResize(object):
    def __init__(self):
        self.p = random.choice([0.5,0.75,1.0,1.5,1.75,2])
    def __call__(self, img, label=None):
        w=img.size[0]*self.p
        h=img.size[1]*self.p
        img = img.resize((w,h),Image.BILINEAR)
        if label:
            label = label.resize((w,h),Image.NEAREST) #left or right
        return img, label
    
    
class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, result):
        if isinstance(result['img'], np.ndarray):
            result['img'] = Image.fromarray(result['img'], mode="RGB")
            if result['label']:
                result['label'] = Image.fromarray(result['label'], mode="L")  

        for a in self.augmentations:
            result = a(result)

        return result

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, label):
        assert img.size == label.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            label.crop((x1, y1, x1 + tw, y1 + th)),
        )


class RandomHorizontallyFlip(object):
    def __call__(self, img, label):
        if random.random() < 0.5:
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                label.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return img, label


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, label):
        assert img.size == label.size
        return (
            img.resize(self.size, Image.BILINEAR),
            label.resize(self.size, Image.NEAREST),
        )


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, label):
        assert img.size == label.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, label
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                label.resize((ow, oh), Image.NEAREST),
            )
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                label.resize((ow, oh), Image.NEAREST),
            )


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, label):
        assert img.size == label.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                label = label.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)

                return (
                    img.resize((self.size, self.size), Image.BILINEAR),
                    label.resize((self.size, self.size), Image.NEAREST),
                )

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, label))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, label):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            img.rotate(rotate_degree, Image.BILINEAR),
            label.rotate(rotate_degree, Image.NEAREST),
        )


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, label):
        assert img.size == label.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, label = (
            img.resize((w, h), Image.BILINEAR),
            label.resize((w, h), Image.NEAREST),
        )

        return self.crop(*self.scale(img, label))