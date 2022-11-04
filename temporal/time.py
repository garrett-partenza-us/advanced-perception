from model import *
import torch
import time
from tqdm import tqdm

BATCH_SIZE = 1
PATCHES = 256
FRAMES = 8
HEIGHT, WIDTH = 400, 400

model = SuperNet(BATCH_SIZE, PATCHES, FRAMES, HEIGHT, WIDTH)

start = time.time()
for img in tqdm(range(10)):
    res = model(torch.rand(BATCH_SIZE,FRAMES,3,HEIGHT,WIDTH))
    print(res.shape)
end = time.time()

print("Total time for ten runs: ", end-start)
