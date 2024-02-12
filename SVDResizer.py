import warnings
warnings.filterwarnings('ignore', module="torchvision")
import ast
import math
import random
import operator as op
import numpy as np

import torch
import torch.nn.functional as F

import torchvision.transforms as T

from nodes import MAX_RESOLUTION, SaveImage
import folder_paths
import comfy.utils

def p(image):
    """Permute the image tensor for processing."""
    return image.permute([0, 3, 1, 2])

def pb(image):
    """Permute the image tensor back to original dimensions."""
    return image.permute([0, 2, 3, 1])

operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,
    ast.USub: op.neg,
    ast.Mod: op.mod,
}

class SVDRsizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 576, "min": 576, "max": 1024, "step": 64, }),
                "height": ("INT", { "default": 1024, "min": 576, "max": 1024, "step": 64, }),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "keep_proportion": ("BOOLEAN", { "default": False }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, width, height, keep_proportion, interpolation="nearest"):
        if keep_proportion is True:
            _, oh, ow, _ = image.shape
            width = ow if width == 0 else width
            height = oh if height == 0 else height
            ratio = min(width / ow, height / oh)
            width = round(ow*ratio)
            height = round(oh*ratio)
        
        outputs = p(image)
        if interpolation == "lanczos":
            outputs = comfy.utils.lanczos(outputs, width, height)
        else:
            outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)
        outputs = pb(outputs)

        return(outputs, outputs.shape[2], outputs.shape[1],)


 
NODE_CLASS_MAPPINGS = {
    "SVDRsizer": SVDRsizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVDResizer": "SVDResizer",
}