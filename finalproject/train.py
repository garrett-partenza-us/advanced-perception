# Garrett Partenza and Jiameng Sung
# CS 7180 Advanced Computer Vision
# December 14th, 2022

# standard
import copy
import os

# external
import pickle
from tqdm import tqdm
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import get_parameters
import tinygrad.nn.optim as optim
import numpy as np
import cv2
from patchify import patchify

# raise error on overflow caused by gelu
np.seterr(all="warn", over="raise")

if not os.path.exists("plots/"):
    os.makedirs("plots")

# MODEL CHOICE
from models.espcnn import Net

# hyperparameters
EPOCHS = 1
BATCH_SIZE = 16
LR = 1e-3

lr_path = "data/LR/{}.png"
hr_path = "data/HR/{}.png"

def get_batch(bs, test=False):
    idxs = np.random.randint(1,81,bs) if not test else np.random.randint(81,100,bs)
    lr = np.array(
        [cv2.imread(lr_path.format(idx)) for idx in idxs]
    )
    hr = np.array(
        [cv2.imread(hr_path.format(idx)) for idx in idxs]
    )
    return lr, hr

# batch img
def batchify(img, bs=8):
    return np.array([img for _ in range(bs)])

# patch img
def patch(batch, bs):
    return np.array([patchify(img, (20,20,3), step=20) for img in batch]).reshape(bs,-1,20,20,3)

# normalize img
def norm(batch):
    return batch/255.0

# mean squared error
def lossfn(p1, p2, p3, t1, t2, t3):
    return t1.sub(p1).square().mean()+t2.sub(p2).square().mean()+t3.sub(p3).square().mean()

# reconstruct image patches
def jigsaw_to_image(x, grid_size=(20, 20)):
    batch_size, num_patches, c, jigsaw_h, jigsaw_w = x.shape
    assert num_patches == grid_size[0] * grid_size[1]
    x_image = x.reshape(batch_size, grid_size[0], grid_size[1], c, jigsaw_h, jigsaw_w)
    output_h = grid_size[0] * jigsaw_h
    output_w = grid_size[1] * jigsaw_w
    x_image = x_image.permute(0, 3, 1, 4, 2, 5).contiguous()
    x_image = x_image.reshape(batch_size, c, output_h, output_w)
    return x_image

# initialize model
net = Net()
optim = optim.Adam(get_parameters(net), LR)
loss_train = []
loss_eval = []

# training loop
for i in tqdm(range(EPOCHS)):
    
    # train mode
    Tensor.traning = True 
    
    # get batch
    x, target = get_batch(BATCH_SIZE)
        
    # build input tensor
    x = Tensor(norm(x), requires_grad=True)
    x.gpu()
    
    # build target tensor
    target = Tensor(norm(target), requires_grad=True) # batch, h, w, c 
    t1 = target[:,:,:,0].reshape(BATCH_SIZE, 384, 384, 1)
    t2 = target[:,:,:,1].reshape(BATCH_SIZE, 384, 384, 1)
    t3 = target[:,:,:,2].reshape(BATCH_SIZE, 384, 384, 1)
    t1.gpu()
    t2.gpu()
    t3.gpu()
        
    # forward pass
    p1, p2, p3 = net.forward(x) # batch, patches, features
    
    # backward pass
    loss = lossfn(p1, p2, p3, t1, t2, t3)
    optim.zero_grad()
    loss.backward()
    optim.clipnorm()
    optim.step()
    
    loss_train.append(loss.data[0])
    
    if i%100==0:

        # build train image
        pred_train = np.concatenate(list(p[0].detach().cpu().data for p in (p1, p2, p3)), axis=-1)
        target_train = target[0].detach().cpu().data
        train_img = np.hstack((pred_train, target_train))
               
        # eval mode
        Tensor.traning = False 
    
        # get batch
        x, target = get_batch(BATCH_SIZE, test=True)

        # build input tensor
        x = Tensor(norm(x), requires_grad=False)
        x.gpu()

        # build target tensor
        target = Tensor(norm(target), requires_grad=False) # batch, h, w, c 
        t1 = target[:,:,:,0].reshape(BATCH_SIZE, 384, 384, 1)
        t2 = target[:,:,:,1].reshape(BATCH_SIZE, 384, 384, 1)
        t3 = target[:,:,:,2].reshape(BATCH_SIZE, 384, 384, 1)
        t1.gpu()
        t2.gpu()
        t3.gpu()
        
        # forward pass
        p1, p2, p3 = net.forward(x) # batch, patches, features

        # backward pass
        loss_test = lossfn(p1, p2, p3, t1, t2, t3)
        
        # log loss
        loss_eval.append(loss_test.data[0])
                
        # build test image
        pred_test = np.concatenate(list(p[0].detach().cpu().data for p in (p1, p2, p3)), axis=-1)
        target_test = target[0].detach().cpu().data
        test_img = np.hstack((pred_test, target_test))
        
        # stack train and test images
        img_stack = np.vstack((train_img, test_img))
        img_stack = (img_stack*255).astype(np.uint8)
        cv2.imwrite("plots/plot_{}.jpg".format(i), img_stack)
        
        del pred_train, target_train, pred_test, target_test, train_img, test_img, img_stack, loss_test
     
    # save 81 image pred every 10 epochs for gif 
    if i%20==0:
        
        Tensor.traning = False 
        
        x = np.array(cv2.imread(lr_path.format(93)))

        # build input tensor
        x = Tensor(norm(batchify(x, bs=1)), requires_grad=False)
        x.gpu()
        
        # forward pass
        p1, p2, p3 = net.forward(x) # batch, patches, features
        
        # build test image
        pred_test = np.concatenate(list(p[0].detach().cpu().data for p in (p1, p2, p3)), axis=-1)
        pred_test = (pred_test*255).astype(np.uint8)
        
        cv2.imwrite("plots/gif_{}.jpg".format(i), pred_test)
        
        del pred_test
        
    # clear cuda memory
    del x, target, p1, p2, p3, t1, t2, t3, loss

# save losses
np.save("losstrain.npy", np.array(loss_train))
np.save("losseval.npy", np.array(loss_eval))

