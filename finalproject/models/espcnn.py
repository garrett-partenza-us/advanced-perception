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
        
        self.cnn1 = (
            Tensor.uniform(64, 3, 5, 5),
            Tensor.zeros(64)
        )
        
        self.cnn2 = (
            Tensor.uniform(64, 64, 3, 3),
            Tensor.zeros(64)
        )
        
        self.cnn3 = (
            Tensor.uniform(3*(4**2), 64, 3, 3),
            Tensor.zeros(3*(4**2))
        )
        
        self.bn1 = BatchNorm2D(64)
        self.bn2 = BatchNorm2D(64)
    
    def phase_shift(self, inpt, scale):
        b, w, h, _ = inpt.shape
        inpt = inpt.permute(1, 2, 3, 0)
        inpt = inpt.reshape(w, h, scale, scale, b)
        inpt = inpt.permute(0,2,1,3,4)
        inpt = inpt.reshape(w*scale, h, scale, b)
        inpt = inpt.reshape(w*scale, h*scale, b)
        inpt = inpt.permute(2, 0, 1)
        return inpt.reshape(b, w * scale, h * scale, 1)

    """
    perform forward pass through conv and transformer
    x: tensor of image patches of shape b,p,h,w,c
    """
    def forward(self, x): # batch, h, w, c
        x = x.permute(0,3,1,2) # batch, c, h, w
        x = self.bn1(x.conv2d(*self.cnn1, stride=1, padding=2).tanh()) # batch, c, h, w
        x = self.bn2(x.conv2d(*self.cnn2, stride=1, padding=1).tanh()) # batch, c, h, w
        x = x.conv2d(*self.cnn3, stride=1, padding=1) # batch, c, h, w
        x = x.permute(0,2,3,1) # batch, h, w, c
        
        scale = 4
        batch, w, h, c = x.shape
        channel_output = c // scale**2 

        x = x.reshape(batch, w, h, channel_output, c//channel_output)
        x = x.permute(0,1,2,4,3)
        x = x.flatten(start_dim=3)

        sections = x.chunk(3, dim=3)
        
        r = self.phase_shift(sections[0], scale=4)
        g = self.phase_shift(sections[1], scale=4)
        b = self.phase_shift(sections[2], scale=4)
                
        return r, g, b
