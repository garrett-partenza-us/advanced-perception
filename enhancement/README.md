Garrett Partenza
September 10th, 2020
CS 7180 Advanced Perception


# Histogram Equalization

### Abstract
This folder contains an implmentation of the image enhancement algorithm "histogram equalization" writtein in C++ using OpenCV. The algorithm is a contrast enhancement algorithm which seeks to more evenly distribute the range of pixel values across the entire image. The algorithm achieves this by constraining the cummulative normalized histogram to be linearly increasing with respect to a parameter K. More can be read on this algorithm \href[https://en.wikipedia.org/wiki/Histogram_equalization][here].

### Introduction
The paper that insipired my implementation is called "A Comparative Study of Histogram Equalization Based Image Enhancement Techniques for Brightjness Preservation and Contrast Enhancement". I like this paper because it discusses the shortcommings of the original algorithm, and proposes new solutions to solve these shortcomings such as brightness preserving bi-histogram equalization (BBHE). While my original submission contains the implementation of the original algorithm, this paper provides the foundation to iterativly reimplment the algorithm in the authors proposed list of new and improved methods.

### Method
The original histogram equalization algorithm for contrast enhancement was implemented in the following process. First, given an image, calculate the histogram of all pixel values. For grayscale images, this can be a single histogram. However, for color images this algorithm works in the same way by treating each RGB channel seperately. Second, normalize the histogram values by dividing by the total number of pixels in the image. This is crucial for the next step, as it ensures the cummulative normalized histogram does not exceed 1. Third, alter the normalized histogram to be cummulative, meaning that the total number of pixels in histogram index 'i' is equal to the total number of pixel values with less than or equal values tom 'i'. Fourth multiply each of the values in the cummulative normalized histogram by L-1, where L is the range of pixel values, normally 256. This step distributres the resulting pixel intensitites across the full intensity range. Finally, transform each pixel in the original image by using our histogram as a look up table. The resulting image will be contrast enhanced.

### Results
Results for two example images can be found in landscape-contrast.jpg and bridge-contrast.jpg. While the algorithm works well for the landscape image, we see that the bridge image is rather dark. This is because the algorithm has a "mean-shift" proble, where it shifts the mean intensity value to the middle gray level of the intensity range. This forms the authors reasoning for proposing new and improved algorithms which attempt to preserve brightness in the original image.

### Reflection
Histogram equalization is a wideley used image enhancement algorithm in fields such as medical imaging. Learning the algorithm provided a good foundation for understanding non-deep image processing techniques. Furthermore, implementing the algorithm from this paper specifically exposed me to the shortcomings of the original algorithm, as well as how those shortcomings could be resolved.  
