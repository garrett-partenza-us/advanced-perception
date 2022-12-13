from tinygrad.tensor import Tensor
from tinygrad.nn import BatchNorm2D
import numpy as np
np.seterr(all='warn', over='raise')

import time

# Transformer class
class TransformerBlock:
    """
    embed_dim: dimension of word embeddings
    num_heads: number of attention heads, must be divisor of embed_dim
    ff_dim : hidden layer dimension of the MLP layer'
    prenorm: whether to apply layernorm before or after attention
    act: activation function
    """
    def __init__(self, embed_dim, num_heads, ff_dim, prenorm=False, act=lambda x: x.relu()):
        self.num_heads = num_heads
        # assert that num_heads divides embed_dim
        self.head_size = embed_dim // num_heads
        assert self.head_size * self.num_heads == embed_dim
        self.prenorm, self.act = prenorm, act
        
        # q, v, k
        self.query = (Tensor.uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
        self.key = (Tensor.uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
        self.value = (Tensor.uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
        
        # final out layer
        self.out = (Tensor.uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

        # MLP layer
        self.ff1 = (Tensor.uniform(embed_dim, ff_dim), Tensor.zeros(ff_dim))
        self.ff2 = (Tensor.uniform(ff_dim, embed_dim), Tensor.zeros(embed_dim))

        # layer normalization layers for attention and activation
        self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
        self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

    def attn(self, x):
        # x: (bs, time, embed_dim) -> (bs, time, embed_dim)
        # apply q, k, v and reshape to bs, time, num_heads, embed_dim/num_heads
        query, key, value = [x.linear(*y) \
          .reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)) \
          for y in [self.query, self.key, self.value]]
                    
        #permute switch time and num_heads
        query = query.transpose(order=(0,2,1,3))  # (bs, num_heads, time, head_size)
        key = key.transpose(order=(0,2,3,1))      # (bs, num_heads, head_size, time)
        value = value.transpose(order=(0,2,1,3))  # (bs, num_heads, time, head_size)

        # get softmax and weighted attention scores
        score = query.dot(key) * (1 / np.sqrt(self.head_size))
                                
        weights = score.softmax()                                   # (bs, num_heads, time, time)
        # new representation is attention * values
        attention = weights.dot(value).transpose(order=(0,2,1,3))   # (bs, time, num_heads, head_size)

        # concat attention from all heads and put through out linear layer
        res = attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out)
        
        return res
    
    def __call__(self, x):
        # whether to apply layer norm before or after attention
        if self.prenorm:
            x = x + self.attn(x.layernorm().linear(*self.ln1)).dropout(0.1)
            x = x + self.act(x.layernorm().linear(*self.ln2).linear(*self.ff1)).linear(*self.ff2).dropout(0.1)
        else:
            # residual plus attention
            x = x + self.attn(x).dropout(0.1)
            # layernorm with learnable linear layer
            x = x.layernorm().linear(*self.ln1)
            # residual plus activation of MLP layer
            x = x + self.act(x.linear(*self.ff1)).linear(*self.ff2).dropout(0.1)
            # ;ayer norm with learnable linear layer
            x = x.layernorm().linear(*self.ln2)
        return x
    
# Vision transformer class
class ViT:
    """
    initialize visual transformer
    layers: number of transformer blocks
    embed_dim = dimensions of linear projection of the image patches
    num_heads : number of attention heads in transformer blocks
    """
    def __init__(self, layers=12, num_heads=3, img_dim=(25,25), out_dim=None, patch_size=25, batch=8, hid_dim=1024):
        
        self.features1 = (
            Tensor.uniform(32, 3, 5, 5),
            Tensor.zeros(64)
        )
        
        self.features2 = (
            Tensor.uniform(64, 32, 3, 3),
            Tensor.zeros(64)
        )
        
        # linear projection of image patches
        self.upsampler1 = (
            Tensor.uniform(32, 32, 9, 9),
            Tensor.zeros(32)
        )

        self.upsampler2 = (
            Tensor.uniform(32, 32, 13, 13),
            Tensor.zeros(32)
        )

        self.upsampler3 = (
            Tensor.uniform(32, 32, 17, 17),
            Tensor.zeros(32)
        )

        self.upsampler4 = (
            Tensor.uniform(32, 32, 19, 19),
            Tensor.zeros(32)
        )
        
        self.out = (
            Tensor.uniform(3, 32, 5, 5),
            Tensor.zeros(3)
        )
        
        self.bn1 = BatchNorm2D(16)
        self.bn2 = BatchNorm2D(32)
        self.bn3 = BatchNorm2D(32)
        self.bn4 = BatchNorm2D(32)
        self.bn5 = BatchNorm2D(32)
        self.bn6 = BatchNorm2D(32)
        
        
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.batch=batch
        self.layers = layers

        # closing statement token (learnable?)
        if not out_dim:
            self.cls = None
        else:
            self.cls = Tensor.ones(1, 1, embed_dim)
        # positional embedding (learnable?)
        if not out_dim:
            self.pos_embedding = Tensor.ones(1, (img_dim[0]*img_dim[1])//(patch_size**2), embed_dim)
        else:
            self.pos_embedding = Tensor.ones(1, (img_dim[0]*img_dim[1])//(patch_size**2)+1, embed_dim)
        # list of transformer blocks with prenorm and GeLU
        
        ## TO DO: GeLU results in overflow from np.exp()
        self.tbs = [
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=hid_dim,
            prenorm=False, act=lambda x: x.sigmoid())
            for i in range(layers)
        ]
        # normalization layer for post transformer
        self.encoder_norm = (Tensor.uniform(embed_dim), Tensor.zeros(embed_dim))
        # prediction layer
        if not out_dim:
            self.head=None
        else:
            self.head = (Tensor.uniform(embed_dim, out_dim), Tensor.zeros(out_dim))

    """
    embed image patches to embed_dim dimensions
    x: tensor of image patches of shape b,p,h,w,c
    """
    def patch_embed(self, x):
        n_patches = x.shape[1]
        x = x.reshape(-1,25,25,3) # batch*patches, p_h, p_w, c
        x = x.permute(0,3,1,2) # batch*patches, c, p_h, p_w
        x = x.conv2d(*self.features1, stride=1, padding=2).relu()
        x = self.bn1(x)
        x = x.conv2d(*self.features2, stride=1, padding=1).relu()
        x = self.bn2(x)
        x = x.conv2d(*self.upsampler1, stride=1, padding=6).relu()
        x = self.bn3(x)
        x = x.conv2d(*self.upsampler2, stride=1, padding=10).relu()
        x = self.bn4(x)
        x = x.conv2d(*self.upsampler3, stride=1, padding=14).relu()
        x = self.bn5(x)
        x = x.conv2d(*self.upsampler4, stride=1, padding=(16,17,16,17))
        x = self.bn6(x)
        x = x.conv2d(*self.out, stride=1, padding=2)
        return x

    """
    perform forward pass through conv and transformer
    x: tensor of image patches of shape b,p,h,w,c
    """
    def forward(self, x): # batch, patches, p_h, p_w, c
        x = self.patch_embed(x) # batch, patches, features
        
        # add cls token if classification
        if self.out_dim:
            ce = self.cls.add(Tensor.zeros(x.shape[0],1,1))
            x = ce.cat(pe, dim=1)
            
        x = x.add(self.pos_embedding) # batch, patches, features
        x = x.sequential(self.tbs) # batch, patches, features
        x = x.layernorm() # batch, patches, features
        x = x.linear(*self.encoder_norm) # batch, patches, features

        # return cls token if classification
        if self.out_dim:
            return x[:, 0].linear(*self.head) # batch, 1, out_dim
        
        return x # batch, patches, features