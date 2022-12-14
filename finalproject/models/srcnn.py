# Garrett Partenza and Jiameng Sung
# CS 7180 Advanced Computer Vision
# December 14th, 2022
from tinygrad.tensor import Tensor
from tinygrad.nn import BatchNorm2D
import numpy as np
np.seterr(all='warn', over='raise')

import time
    
# Vision transformer class
class Net:
    """
    initialize visual transformer
    layers: number of transformer blocks
    embed_dim = dimensions of linear projection of the image patches
    num_heads : number of attention heads in transformer blocks
    """
    def __init__(self):
        

        self.kernel = (
            Tensor.uniform(64, 3, 5, 5),
            Tensor.zeros(64)
        )

        
        self.kernel2 = (
            Tensor.uniform(32, 64, 5, 5),
            Tensor.zeros(32)
        )

        
        self.kernel3 = (
            Tensor.uniform(3, 32, 3, 3),
            Tensor.zeros(3)
        )

        self.bn1 = BatchNorm2D(64)
        self.bn2 = BatchNorm23(32)
                               
    
    def upsample(self, x):
        bs,c,py,px = x.shape
        x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 4, px, 4).reshape(bs, c, py*4, px*4)
        return x

    """
    perform forward pass through conv and transformer
    x: tensor of image patches of shape b,p,h,w,c
    """
    def forward(self, x): # batch, h, w, c
        x = x.permute(0,3,1,2) # batch, c, h, w
        x = self.upsample(x)
        x = x.conv2d(*self.kernel1, stride=1, padding=2).relu() # batch, c, h, w
        x = self.bn1(x)
        x = x.conv2d(*self.kernel2, stride=1, padding=2).relu() # batch, c, h, w
        x = self.bn2(x)
        x = x.conv2d(*self.kernel3, stride=1, padding=1) # batch, c, h, w
        x = x.permute(0,2,3,1) # batch, h, w, c                
        return x.chunk(3, dim=3)
