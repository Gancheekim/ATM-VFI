"""A VGG-based perceptual loss function for PyTorch."""
"""
code borrowed from https://github.com/crowsonkb/vgg_loss
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.models import VGG16_Weights
from torchvision.models import VGG19_Weights
import torchvision

class Lambda(nn.Module):
    """Wraps a callable in an :class:`nn.Module` without registering it."""

    def __init__(self, func):
        super().__init__()
        object.__setattr__(self, 'forward', func)

    def extra_repr(self):
        return getattr(self.forward, '__name__', type(self.forward).__name__) + '()'


class WeightedLoss(nn.ModuleList):
    """A weighted combination of multiple loss functions."""

    def __init__(self, losses, weights, verbose=False):
        super().__init__()
        for loss in losses:
            self.append(loss if isinstance(loss, nn.Module) else Lambda(loss))
        self.weights = weights
        self.verbose = verbose

    def _print_losses(self, losses):
        for i, loss in enumerate(losses):
            print(f'({i}) {type(self[i]).__name__}: {loss.item()}')

    def forward(self, *args, **kwargs):
        losses = []
        for loss, weight in zip(self, self.weights):
            losses.append(loss(*args, **kwargs) * weight)
        if self.verbose:
            self._print_losses(losses)
        return sum(losses)


class TVLoss(nn.Module):
    """Total variation loss (Lp penalty on image gradient magnitude).

    The input must be 4D. If a target (second parameter) is passed in, it is
    ignored.

    ``p=1`` yields the vectorial total variation norm. It is a generalization
    of the originally proposed (isotropic) 2D total variation norm (see
    (see https://en.wikipedia.org/wiki/Total_variation_denoising) for color
    images. On images with a single channel it is equal to the 2D TV norm.

    ``p=2`` yields a variant that is often used for smoothing out noise in
    reconstructions of images from neural network feature maps (see Mahendran
    and Vevaldi, "Understanding Deep Image Representations by Inverting
    Them", https://arxiv.org/abs/1412.0035)

    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.
    """

    def __init__(self, p, reduction='mean', eps=1e-8):
        super().__init__()
        if p not in {1, 2}:
            raise ValueError('p must be 1 or 2')
        if reduction not in {'mean', 'sum', 'none'}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.p = p
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target=None):
        input = F.pad(input, (0, 1, 0, 1), 'replicate')
        x_diff = input[..., :-1, :-1] - input[..., :-1, 1:]
        y_diff = input[..., :-1, :-1] - input[..., 1:, :-1]
        diff = x_diff**2 + y_diff**2
        if self.p == 1:
            diff = (diff + self.eps).mean(dim=1, keepdims=True).sqrt()
        if self.reduction == 'mean':
            return diff.mean()
        if self.reduction == 'sum':
            return diff.sum()
        return diff


class VGGLoss(nn.Module):
    """Computes the VGG perceptual loss between two batches of images.

    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
    normalized to the range 0–1.

    The VGG perceptual loss is the mean squared difference between the features
    computed for the input and target at layer :attr:`layer` (default 8, or
    ``relu2_2``) of the pretrained model specified by :attr:`model` (either
    ``'vgg16'`` (default) or ``'vgg19'``).

    If :attr:`shift` is nonzero, a random shift of at most :attr:`shift`
    pixels in both height and width will be applied to all images in the input
    and target. The shift will only be applied when the loss function is in
    training mode, and will not be applied if a precomputed feature map is
    supplied as the target.

    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.

    :meth:`get_features()` may be used to precompute the features for the
    target, to speed up the case where inputs are compared against the same
    target over and over. To use the precomputed features, pass them in as
    :attr:`target` and set :attr:`target_is_features` to :code:`True`.

    Instances of :class:`VGGLoss` must be manually converted to the same
    device and dtype as their inputs.
    """

    models = {'vgg16': models.vgg16, 'vgg19': models.vgg19}

    def __init__(self, model='vgg16', layer=None, shift=0, reduction='mean', loss='l1_loss', do_normalize=False):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.do_normalize = do_normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        # self.model = self.models[model](weights=VGG16_Weights.DEFAULT).features[:layer+1]
        self.model = self.models[model](weights=VGG16_Weights.DEFAULT).features
        print(self.model)
        self.model.eval()
        self.model.requires_grad_(False)
        if loss == "l1_loss":
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.MSELoss()

    def get_features(self, input):
        if self.do_normalize:
            return self.model(self.normalize(input))
        return self.model(input)

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            sep = input.shape[0]
            batch = torch.cat([input, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            feats = self.get_features(batch)
            input_feats, target_feats = feats[:sep], feats[sep:]
        return self.loss(input_feats, target_feats)
    

'''
code borrowed from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
'''    
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False, vgg_type=16, do_normalize=True, 
                 use_perceptual_loss=True, use_style_loss=True,
                 perceptual_criterion=nn.L1Loss(), style_criterion=nn.MSELoss()):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        if vgg_type == 16:
            blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features[:4].eval())
            blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features[4:9].eval())
            blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features[9:16].eval())
            blocks.append(torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features[16:23].eval())
        elif vgg_type == 19:
            blocks.append(torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).features[:4].eval())
            blocks.append(torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).features[4:9].eval())
            blocks.append(torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).features[9:18].eval())
            blocks.append(torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).features[16:27].eval())
            blocks.append(torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).features[27:36].eval())

        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.do_normalize = do_normalize
        self.use_perceptual_loss = use_perceptual_loss
        self.use_style_loss = use_style_loss
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.perceptual_criterion = perceptual_criterion
        self.style_criterion = style_criterion

    def forward(self, input, target):
        target = target.detach()
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        if self.do_normalize:
            input = (input-self.mean) / self.std
            target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        # perceptual_loss = torch.zeros().to(self.mean.device)
        # style_loss = torch.zeros().to(self.mean.device)
        perceptual_loss = 0.
        style_loss = 0.
        x = input
        y = target.detach()
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if self.use_perceptual_loss:
                perceptual_loss += self.perceptual_criterion(x, y)
            if self.use_style_loss:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                style_loss += self.style_criterion(gram_x, gram_y)
        return perceptual_loss, style_loss