# Garrett Partenza and Jamie Sun
# November 5, 2022
# CS Advanced Perception


# Imports 
import torch
from torch import nn
from einops import rearrange
import math
import time
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# feed forward class for transformer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, last):
        super().__init__()
        if not last:
            last=dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, last),
        )
        
    def forward(self, x):
        return self.net(x)

    
# attention class for transformer
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# transformer class
class Transformer(nn.Module):
    def __init__(self, batch_size, patches, frames, dim, depth, heads, dim_head, mlp_dim, out_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for d in range(depth):
            if d == depth-1:
                last=out_dim
            else: 
                last=None
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim, last)
            ]))
        self.batch_size = batch_size
        self.patches = patches
        self.frames = frames
        self.dim = dim
        self.out_dim = out_dim
            
    def forward(self, x):
        for d in range(self.depth):
            attn, ff = self.layers[d]
            x = torch.mul(nn.Softmax(dim=2)(attn(x)), x) + x
            x = ff(x)
            if d!=self.depth-1:
                x+=x
        return x
    
    
# convnet class
class ConvNet(nn.Module):
    def __init__(self, batch_size, patches, frames, patch_size):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 128, 3, 3)
        self.conv3 = nn.Conv2d(128, 256, 3, 3)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.batch_size = batch_size
        self.patches = patches
        self.frames = frames
        self.patch_size = patch_size
        
    def forward(self, x):
        x = x.reshape(self.batch_size*self.frames*self.patches, 3, self.patch_size, self.patch_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(start_dim=1)
        x = x.reshape(self.batch_size, self.frames, self.patches, 1024)
        return x
    
        
# our model class (combined convnet and transformer with reshape)
class SuperNet(nn.Module):
    def __init__(self, batch_size, patches, frames, width, height, blocks=6):
        super(SuperNet, self).__init__()
        self.batch_size = batch_size
        self.patches = patches
        self.frames = frames
        self.width = width
        self.height = height
        self.patch_size = int(math.sqrt((width*height)/patches))
        self.blocks = blocks
        assert (width*height)%self.patch_size == 0
        self.convnet = ConvNet(
            batch_size, 
            patches, 
            frames,
            self.patch_size
        )
        self.transformer = Transformer(
            batch_size,
            patches,
            frames,
            dim=1024,
            depth=blocks,
            heads=8,
            dim_head=128,
            mlp_dim=2048,
            out_dim=1536
        )

    def forward(self, x):
        # B F P 3 25 25
        x = self.convnet(x)
        # B F P feat
        x = x.permute(0,2,1,3)
        # B P F feat
        x = x.reshape(self.batch_size*self.patches, self.frames, 1024)
        # B*P F 1024
        x = self.transformer(x)
        # B*P F 1536
        x = x.reshape(self.batch_size, self.patches, self.frames, 1536)
        # B P F 1536
        x = x.permute(0,2,1,3)
        # B F P 1536
        x = x.reshape(self.batch_size, 3, 1024, 1024)
        return x
    
    def plot(self, x, title):
        x = make_grid(x, normalize=True, scale_each=True)
        x = x.cpu().clone().detach()
        if x.dtype == torch.uint8:
            x = x / 255.0
        if x.is_floating_point():
            if x.ndim == 3 and x.shape[0] >= 3:
                # Channels first to channels last [1,2,3] -> [3,2,1]
                x = x.permute(1, 2, 0)
                plt.imshow(x, vmin=0, vmax=1, interpolation="nearest")
            else:
                x = x[0]
                plt.imshow(x)
        plt.savefig("plots/{}.png".format(title))
        plt.clf()