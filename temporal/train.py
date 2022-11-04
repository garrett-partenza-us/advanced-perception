from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.datasets import *
from model import *

transforms = make_transforms_JIF(lr_bands_to_use='true_color', radiometry_depth=12)
multiprocessing_manager = Manager()

dataset_root = '/scratch/partenza.g/'
hr_dataset_folder = 'hr_dataset/12bit/'
lr_dataset_folder = 'lr_dataset/'

BATCH_SIZE = 8
HOLDOUT = 0.2
PATCHES = 256
FRAMES = 8
WIDTH, HEIGHT = 400, 400


def generate_dataloaders():
    
    lr = SatelliteDataset(
        root=os.path.join(dataset_root, lr_dataset_folder, "ASMSpotter*", "L2A", ""),
        file_postfix="-L2A_data.tiff",
        transform=transforms["lr"],
        number_of_revisits=8,
        bands_to_read=SPOT_RGB_BANDS,
        multiprocessing_manager=multiprocessing_manager
    )

    hr_rgb = SatelliteDataset(
        root=os.path.join(dataset_root, hr_dataset_folder, "ASMSpotter*"),
        file_postfix="_rgbn.tiff",
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
    
    train, temp = train_test_split(range(len(dataset)), test_size=HOLDOUT)
    val, test = train_test_split(temp, test_size=0.5)
    train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    

def main():
    
    train, val, test = generate_dataloaders()
    
    model = SuperNet(BATCH_SIZE, PATCHES, FRAMES, WIDTH, HEIGHT)
    
    # run train loop
    
if __name__=="__main__":
    main()
