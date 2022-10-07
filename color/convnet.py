import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import numpy as np
import torch
import os
import pandas as pd

BATCH_SIZE = 64

with open('img_tensors.pickle', 'rb') as handle:
    img_tensors = pickle.load(handle)
    
df = pd.read_csv("SimpleCube++/train/gt.csv")

labels = []

for img in os.listdir("SimpleCube++/train/processed"):
    row = df[df["image"]==img.split(".")[0].strip()] 
    label = [row.mean_r.iloc[0], row.mean_g.iloc[0]]
    labels.append(label)

labels = np.array(labels)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(img_tensors, labels, test_size=0.2)

loader_train = DataLoader(
    list(zip(x_train, y_train)),
    shuffle=True,
    drop_last=True,
    batch_size=BATCH_SIZE
)

loader_test = DataLoader(
    list(zip(x_test, y_test)),
    shuffle=True,
    drop_last=True,
    batch_size=BATCH_SIZE
)

import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 11, 1)
        self.conv2 = nn.Conv2d(64, 128, 7, 3)
        self.conv3 = nn.Conv2d(128, 256, 11, 1)
        self.dropout = nn.Dropout(0.2)
        self.lin1 = nn.Linear(4096,256)
        self.lin2 = nn.Linear(256,2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x 
    
from tqdm import tqdm

model = ConvNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
loss_func = torch.nn.MSELoss()

EPOCHS = 100

losses = []

for epoch in range(EPOCHS):
    
    epoch_loss = []
            
    for batch in loader_train:
         
        optimizer.zero_grad()
        x, y = batch 
        x, y = x.to(device), y.to(device) 
        loss = loss_func(model(x.float()), y.float())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        epoch_loss.append(loss.item())

    print("Epoch {}: MSE: {}".format(epoch, loss))
        
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plt.title("Training Loss (MLP)")
plt.plot(losses)
plt.savefig("convnet.png")
