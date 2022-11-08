# Garrett Partenza and Jamie Sun
# November 5, 2022
# CS Advanced Perception


# Imports  
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.datasets import *
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.transforms as T
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, InterpolationMode, Normalize, Lambda
from helper import *


# speed up training with benchmarking
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# generate required transforms
y_transform  = make_transforms_JIF()
lr_resize = T.Resize((400,400))
hr_resize = T.Resize((1024,1024))
transforms = make_transforms_JIF(lr_bands_to_use='true_color', radiometry_depth=12)

# implement multiprocessing
multiprocessing_manager = Manager()

# file system paths to dataset
dataset_root = '/scratch/partenza.g/'
hr_dataset_folder = 'hr_dataset/12bit/'
lr_dataset_folder = 'lr_dataset/'

# hyperparamters
BATCH_SIZE = 8
HOLDOUT = 0.2
PATCHES = 256
FRAMES = 8
WIDTH, HEIGHT = 400, 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30


# function to plot and save generated images on test set every epoch
def plot(x, epoch):
    x = make_grid(x, normalize=True, scale_each=True)
    x = x.clone().detach()
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
    plt.savefig("plots/epoch_{}.png".format(epoch))
    plt.clf()
    

# function to generate index dataloaders for all illegal mining images
def generate_dataloaders(batch_size=8):
    
    lr = SatelliteDataset(
        root=os.path.join(dataset_root, lr_dataset_folder, "ASMSpotter*", "L2A", ""),
        file_postfix="-L2A_data.tiff",
        transform=transforms["lr"],
        number_of_revisits=8,
        bands_to_read=S2_ALL_12BANDS["true_color"],
        multiprocessing_manager=multiprocessing_manager
    )

    hr_rgb = SatelliteDataset(
        root=os.path.join(dataset_root, hr_dataset_folder, "ASMSpotter*"),
        file_postfix="_ps.tiff",
        transform=transforms["hr"],
        bands_to_read=SPOT_RGB_BANDS,
        number_of_revisits=1,
        multiprocessing_manager=multiprocessing_manager
    )
    
    dataset = DictDataset(
        **{
            "lr": lr,
            "hr": hr_rgb,
        }
    )
    
    train, temp = train_test_split(list(range(len(dataset))), test_size=HOLDOUT)
    val, test = train_test_split(temp, test_size=0.5)
    np.save("split/train.npy", np.array(train))
    np.save("split/val.npy", np.array(val))
    np.save("split/test.npy", np.array(test))
    train = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val = DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    test = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    return dataset, train, val, test
    

def patch(x):
    kernel_size, stride = 25, 25
    x = x.unfold(1, 25, 25).unfold(2, 25, 25)
    x = x.contiguous().view(x.size(0), -1, 25, 25).permute(1,0,2,3)
    return x
    
    
# main training loop
def main():
    
    #initialize objects
    print("Generating dataloaders...")
    dataset, train, val, test = generate_dataloaders(batch_size=BATCH_SIZE)
    print("Train ({}), Val ({}), Test ({})".format(len(train)*BATCH_SIZE, len(val)*BATCH_SIZE, len(test)*BATCH_SIZE))
    print("Initializing model parameters...")
    model = SuperNet(BATCH_SIZE, PATCHES, FRAMES, WIDTH, HEIGHT, blocks=8).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    loss_func = torch.nn.MSELoss()
    scheduler = ExponentialLR(optimizer, gamma =0.95)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable paramters: {}".format(params))
    print("\n\n", "*"*30, "\n\n", "Begging training\n\n", "*"*30, "\n\n")
    
    # train
    loss_train, loss_val = [], []
    for epoch in range(EPOCHS):
        temp = []
        for batch in tqdm(train):
            optimizer.zero_grad(set_to_none=True)
            x = torch.stack([lr_resize(dataset[idx]['lr']) for idx in batch])
            x = torch.stack(
            [
                patch(img) for img in x.flatten(end_dim=1)
            ]).reshape(BATCH_SIZE, FRAMES, PATCHES, 3, 25, 25)
            x.requires_grad=True
            x = x.to(DEVICE)
            y = torch.stack(
                [hr_resize(y_transform['hr'].transforms[1](dataset[idx]['hr'])) for idx in batch]
            ).flatten(end_dim=1)
            y.requires_grad=True
            y = y.to(DEVICE)
            with torch.autocast(device_type="cuda"):
                out = model(x.float())
                loss = loss_func(out, y.float())
            loss.backward()
            optimizer.step()
            
            # free memory
            x.detach().cpu()
            y.detach().cpu()
            out.detach().cpu()
            loss.detach().cpu()
            loss_train.append(loss.item())
            temp.append(loss.item())
            del x, y, out, loss
            torch.cuda.empty_cache()

        scheduler.step()
        
        if epoch%5==0:
            torch.save(model.state_dict(), "models/epoch_{epoch}.pth".format(epoch=epoch))
            torch.save(optimizer.state_dict(), "models/epoch_{epoch}.pth".format(epoch=epoch))
        print("Epoch {}: Train MSE: {}".format(epoch, torch.mean(torch.tensor(temp))))
        del temp
        
        # validation
        temp = []
        with torch.no_grad():
            for batch in val:
                optimizer.zero_grad(set_to_none=True)
                x = torch.stack([lr_resize(dataset[idx]['lr']) for idx in batch])
                x = torch.stack(
                [
                    patch(img) for img in x.flatten(end_dim=1)
                ]).reshape(BATCH_SIZE, FRAMES, PATCHES, 3, 25, 25)
                x = x.to(DEVICE)
                y = torch.stack(
                    [hr_resize(y_transform['hr'].transforms[1](dataset[idx]['hr'])) for idx in batch]
                ).flatten(end_dim=1)
                y = y.to(DEVICE)
                out = model(x.float())
                loss = loss_func(out, y.float())
                
                # free memory
                x.detach().cpu()
                y.detach().cpu()
                out.detach().cpu()
                loss.detach().cpu()
                temp.append(loss.item())
                loss_val.append(loss.item())
                del x, y, out, loss
                torch.cuda.empty_cache()
                
        # generate image from test
        with torch.no_grad():
            for batch in test:
                optimizer.zero_grad(set_to_none=True)
                x = torch.stack([lr_resize(dataset[idx]['lr']) for idx in batch])
                x = torch.stack(
                [
                    patch(img) for img in x.flatten(end_dim=1)
                ]).reshape(BATCH_SIZE, FRAMES, PATCHES, 3, 25, 25)
                x = x.to(DEVICE)
                out = model(x.float())[0]
                x.detach().cpu()
                out.detach().cpu()
                plot(out.detach().cpu(), epoch)
                # free memory
                del x, out
                torch.cuda.empty_cache()
                break

        print("Epoch {}: Test MSE: {}".format(epoch, torch.mean(torch.tensor(temp))))
        del temp
        
    np.save("models/training_loss.npy", np.array(loss_train))
    np.save("models/validation_loss.npy", np.array(loss_val))
    
    
if __name__=="__main__":
    main()
