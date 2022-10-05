import cv2
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

#path to SimpleCube dataset
dataset_path = "SimpleCube++/"
#rgb conversion matrix
cam2rgb = np.array([
        1.8795, -1.0326, 0.1531,
        -0.2198, 1.7153, -0.4955,
        0.0069, -0.5150, 1.5081,]).reshape((3, 3))


def linearize(img, black_lvl=2048, saturation_lvl=2**14-1):
    """
    remove black level and saturation according to paper
    """
    return np.clip((img - black_lvl)/(saturation_lvl - black_lvl), 0, 1)


def adjust(cam, img, ilum):
    """
    adjust the illumination to cannonical ground truth
    gt[gt["image"]==img.split(".")[0]].to_numpy().flatten()[1:]
    """
    return np.clip(cam/illum, 0, 1)

    
def to_rgb(cam):
    """
    https://github.com/Visillect/CubePlusPlus/blob/master/challenge/make_preview.py
    """
    cam = np.dot(cam, cam2rgb.T)
    return np.clip(cam, 0, 1)**(1/2.2)


def denormalize(cam):
    """
    scale from 0-1 range to 0-255 range
    """
    return (cam*255).astype(np.uint8)


def binarize(cam, steps=64):
    """
    convert a normalized rgb image to its binarized histogram
    according to the papers sampling method
    """
    #flatten rows and columns
    cam = cam.reshape(cam.shape[0]*cam.shape[1], -1)
    #list for tracking rg chromaticity coordinates
    rg_coords = []
    #iterate over pixels and get coordinates
    for pix_idx in range(cam.shape[0]):
        r = cam[pix_idx][0]
        g = cam[pix_idx][1]
        rg_coords.append((r,g))
    #convert from list to dataframe
    rg_df = pd.DataFrame(rg_coords, columns=["r", "g"])
    #bin coordinates according to step size
    rg_df = rg_df.assign(
        r_cut=pd.cut(rg_df.r, np.sqrt(steps).astype(int), labels=list(range(0,np.sqrt(steps).astype(int)))),
        g_cut=pd.cut(rg_df.g, np.sqrt(steps).astype(int), labels=list(range(0,np.sqrt(steps).astype(int))))
    )
    #zip and reduce to set
    step_coords = list(zip(rg_df.r_cut, rg_df.g_cut))
    step_coords = set(step_coords)
    #initialize grid of zeros
    hist = np.zeros((np.sqrt(steps).astype(int), np.sqrt(steps).astype(int)))
    #turn on signal for present coordiniates
    for x in range(np.sqrt(steps).astype(int)):
        for y in range(np.sqrt(steps).astype(int)):
            if (x,y) in step_coords:
                hist[x][y]=1  
    return hist.flatten()


def main():
    """
    iterate over all images and save binarized histograms
    into a pandas dataframe
    """
    hists = []
    for img in tqdm(os.listdir(dataset_path+"train/PNG")):
        try:
            #read image
            cam = cv2.imread(dataset_path+"train/PNG/"+img, cv2.IMREAD_UNCHANGED)
            #bgr -> rgb
            cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
            #uint16 -> float64
            cam = cam.astype(np.float64)
            #black level and saturation
            cam = linearize(cam)
            #not sure what this step means
            cam = to_rgb(cam)
            cv2.imwrite(dataset_path+"train/processed/{}".format(img), cam)
            #compute binary histogram
#             hist = binarize(cam, steps=64)
#             hists.append((img, hist))
        except:
            print("failed for {}".format(img))
    
if __name__=="__main__":
    main()
