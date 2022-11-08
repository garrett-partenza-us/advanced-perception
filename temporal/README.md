# Learning Multi-Frame Super Resolution via Patch-Attention

[Wiki Report](https://wiki.khoury.northeastern.edu/display/~mucc001/CS+7180+Assignment+3%3A+Transformer-based+network+for+multiple+images+Super-resoluton)

Garrett Partenza and Jamie Sun
November 5, 2022
CS Advanced Perception

* It may not be the best super-resolution model, but its a super-resolution model :) *

### Codebase Directory: Explanation of files along with how to execute.
1. requirements.txt - pip library requirements file
2. environment.yaml - conda environment install file
3. model.py - PyTorch implementation of our model
4. train.py - script to load model and execute training on the worldstrat dataset
5. plots - generating high resolution images from our model durring training
6. src - src code from the worldstrat github repository to help load the image dataset
7. helper.py - helper functions for our training script
8. split - folder containing numpy arrays of the train/val/test split indicices
9. models - folder containing the saved model weights

To execute the code, install the required libraries (either using pip or conda) and download the worldstrat dataset into your scratch directory on the HPC. We had to use this directory due to the large file size of the dataset (over 100 GBs). Next, change the directory variables in train.py to match your file system. After completing these steps, you can excute the train.py script. All HPC machines used in this research were linux-based.

### Travel Days:
We would like to use one of our travel days for this assignment, as we may have bit off more than we could chew in a one-week time span.

### Abstract: An abstract or summary of your project (200 words or less)
For assignment 3, we designed and implemented a CNN and Transformer architecture for Multi-Temporal images super-resolution. We work with WorldStrat Dataset which contains multi-temporal low-resolution satellite images and the corresponding high-resolution satellite image. The input to our model is 8 low-resolution satellite images (400 * 400) for the same scene taken at different times and the output to our model is one high-resolution image (1024 * 1024) of the scene. The model starts with a CNN component which splits each low-resolution image into 256 patches and put the patches into the transformer component. The transformer component has one multi-head attention module and combines the patches produced by the multi-head attention module together to form a high-resolution image. We were not able to produce an ideal result but will improve the model for our final project.

### Introduction and prior work: A brief introduction that includes a description of the paper that inspired your project and other directly relevant resources you used.
High-resolution satellite images are costly to produce and hard to access. Therefore, it is important to produce high-resolution satellite images based on the more accessible low-resolution images. In this project, we use the WorldStrat dataset created by Cornebise et al.(2022) to tackle the problem of super-resolution for low-resolution satellite images. The authors of the WorldStrat dataset choose three benchmark models for multitemporal images super-resolution for their dataset. The single-image super-resolution architecture SRCNN by Dong et al., 2015, a multi-frame extension of SRCNN by collating revisits as channels created by the authors, and a multi-spectral modification of the original HighResNet (Deudon et al., 2020) to handle multiple bands similarly to (Razzak et al., 2021). To inform our design which has a transformer component, we also read the paper A Hybrid Network of CNN and Transformer for Lightweight Image Super-Resolution. In the paper. The authors propose a hybrid model of CNN and Transformer for lightweight image super-resolution (HNCT). The model consists of four parts: shallow feature extraction module, hybrid blocks of CNN and Transformer(HBCT), dense feature fusion module, and up-sampling module. In this project, we also propose a hybrid model of CNN and Transformer but for super-resolution of multi-temporal low-resolution satellite images. Our model consists of a CNN model used for feature extraction, and a transformer module taking care of the up-sampling.

### Methods: A brief description of your process and algorithm(s)
Out model consists of both a convolutional model and a trasnformer model. The input to the model is a sequence of 8 low resolution satelite images, and the output is the corresponding high resolution image. Our methodology first chunks the low resoltuion image into patches, then vectorizes the patches into a single feature vector using a conventional convolutional neural network. Then, we perform attention between the corresponding patches across the eight frames. The idea here is that each image patch contains some portion of information needed to produce the higher resolution patch, and thus multi-headed attention can facilitate in extracting this information between the frames. Finally, the two-dimensional output of the last transformer block is reshaped into the three-dimensional channel high resolution image. Our model had 39,175,424 trainable paramters and was trained on a T4 GPU using the Northeastern HPC. Our loss function was the mean squared error between the predicted image and ground truth. 

### Dataset
The dataset chosen for this project is the WorldStrat dataset created by Cornebise et al.(2022). The dataset consists of nearly 10,000 km² of free high-resolution and matched low-resolution satellite imagery of unique locations. The dataset contains Airbus SPOT 6/7 satellites’ high-resolution images of up to 1.5 m/pixel and multiple low-resolution images from the freely accessible Sentinel-2 satellites at 10 m/pixel, which are temporally matched with each high-resolution image. The dataset makes a stratified representation of all types of land use across the world: from agriculture to ice caps, from forests to multiple urbanization densities, and assigns labels for locations such as sites of humanitarian interest, illegal mining sites, and settlements of persons at risk. Due to computational constraints, we selected a subset of the dataset which images suspected illegal mining sites. More reguarding the dataset can be found on their [homepage](https://zenodo.org/record/6810792#.Y2gCzuzMLBc). 

### Results: Show the result of your work on multiple appropriate images, showing both the strengths and weaknesses of the method. Include a short description with each result image indicating anything special about it.
Our model resulted in overall poor performance and was unable to produce reasonable imagery. However, we attribute this to time constraints, as we spent most of our time developing the training pipeline. As a result, we were only able to train the model once, and did not begin attempting to conduct hyperparameter tuning or model debugging. We hypothesize that, due to the large number of tensor reshapes in our model, there is a bug causing either spatial or temporal information to be lost. 

### Reflection and acknowledgements: reflect on what you learned and acknowledge any assistance or resources you received or used during the project.
While the results of our model were poor, we learned a lot about transformer models and how they relate to sequence data. More specifically, we built out model in pytorch from scratch, and even had to customize the last encoder block for our needed 2D-to-3D reshape step. Furthermore, this project exposed us to dealing with memory-intensive datasets, and forced us to be clever in building custom dataloaders and implementing other efficiency improvment methods in our training pipeline. We propose to debug and correct our model for the final project, and if possible, extend our model to augment existing object recongition systems using satalite imagery. This is becauise we believe the augmentation of super resolution will improve the performance of existing geospatial object recognition tasks. 
