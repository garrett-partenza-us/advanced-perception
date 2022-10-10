#### Garrett Partenza and Jiameng Sun
#### October 11, 2022
#### CS7180 Advanced Perception

### Abstract
Inspired by Estimating the scene illumination chromaticity by using a neural network, we implement a multilayer neural network that recovers the illumination chromaticity with only an input image. The neural network consists of two layers of perceptrons and is trained with a database for color constancy called SimpleCube++. The illumination chromaticity is used to render the image under neutral light with the diagonal model. Except for implementing the same neural network presented in the paper (original_net), we created a modern net with similar two layers of perceptrons but uses ReLU instead of Sigmoid as the activation function and uses mean square error instead of Euclidean distance as the error metric. Finally, we created a CNN network trained with the same dataset for the same task as the perceptron networks. 

### Introduction
The paper that inspired our implementation is Estimating the scene illumination chromaticity by using a neural network. It uses a very simple neural network that contains only two layers of perceptrons but obtains pretty good estimates of illumination chromaticity. The authors also use rg chromaticity histogram as the input for their neural network and claim that the intensity of the rg chromaticity and the spatial information is less relevant than the rg chromaticity histogram. In this project, we not only implement the authors’ perceptron network but also create a CNN network to test the authors’ claim. 

### Method
In this project, we replicate the paper's multi-layer perceptron architecture, as well as implement two more architectures – (1) a modern MLP with ReLU activation functions and Adam optimizer, and (2) a convolutional neural networkBoth multi-layer perceptron architectures take the sampled rg color-space histogram as input, while the convolutional neural network takes in ta downsampled version of the entire image. The dataset used to test all three models is the SimpleCube++ dataset, containing ~2000 images with ground truth color constancy levels. More regarding the dataset can be found in their [arxiv preprint](https://ieeexplore.ieee.org/document/9296220) or [GitHub project page](https://github.com/Visillect/CubePlusPlus). 

### Results
The architecture replicating the original paper achieves a final mean-squared error of 0.0008. On the other hand, the modern MLP achieves a final test error of 0.0005, outperforming the original architecture. The convolutional neural net fails to learn any correlation between the image and color constancy even with a rather large parameter count of 5,439,362 between both the convolutional and linear layers. The convolutional neural network fails to converge steadily even when assisted by an exponential learning rate scheduler. We estimate that, due to the higher number of parameters, the convolutional neural network requires more data than the MLP architectures. All relevant code, including the loss curves for all three models, may be found in our assignment’s GitHub repository. 

### Reflection
More information does not necessarily provide better performance. Though the perceptron gets only an rg chromaticity histogram as the input, it performs better than the CNN which not only gets additional RGB intensity information but also spatial information. We hypothesize that the poor performance of a simple CNN network encouraged Shi et al. to propose a novel structure of HypNet and SelNet in Deep Specialized Network for Illuminant Estimation. 

### Instructions for Running
The code for both multi-layer perceptron models are contained in a Jupyter Notebook file, so execution and imports should be fairly straightforward. The Convolutional neural network, being a large model, required GPU training and thus is stored in a normal python file called convnet.py. Loss curves for the MLP models may be found inside the Jupyter Notebook files, while the convolutional neural network’s loss curve is stored in convnet.png. Due to GitHub large file restrictions, the SimpleCube++ dataset was not able to be pushed to the repository. However, their project GitHub contains the wget command for download.

