Garrett Partenza and Jamie Sun
September 10th, 2020
CS 7180 Advanced Perception


# Histogram Equalization

### Link to Wiki report
https://wiki.khoury.northeastern.edu/display/~mucc001/CS7180%3A+Image+Enhancement+with+Histogram+Equalization (It has the same content, but a better display of images in the result section.)

### Abstract
This folder contains an implementation of the image enhancement algorithm "histogram equalization" written in C++ using OpenCV. The algorithm is a contrast enhancement algorithm that seeks to more evenly distribute the range of pixel values across the entire image. The algorithm achieves this by constraining the cumulative normalized histogram to be linearly increasing with respect to a parameter K. More can be read on this algorithm \href[https://en.wikipedia.org/wiki/Histogram_equalization][here].
In addition, the folder includes an implementation of Brightness Preserving Bi-Histogram Equalization (BBHE) and an implementation of Equal Area Dualistic Sub-Image Histogram Equalization (DSIHE). Both algorithms are based on histogram equalization. 

### Introduction
The paper that inspired our implementation is called "A Comparative Study of Histogram Equalization Based Image Enhancement Techniques for Brightjness Preservation and Contrast Enhancement". We like this paper because it discusses the shortcomings of the original algorithm, and proposes new solutions to solve these shortcomings such as brightness preserving bi-histogram equalization (BBHE). Our submission contains not only the implementation of the original algorithm but also two algorithms in the authors' list of newer and improved methods.


### Method
The original histogram equalization algorithm for contrast enhancement was implemented in the following process. First, given an image, calculate the histogram of all pixel values. For grayscale images, this can be a single histogram. However, for color images, this algorithm works in the same way by treating each RGB channel separately. Second, normalize the histogram values by dividing them by the total number of pixels in the image. This is crucial for the next step, as it ensures the cumulative normalized histogram does not exceed 1. Third, alter the normalized histogram to be cumulative, meaning that the total number of pixels in the histogram index 'i' is equal to the total number of pixel values with less than or equal values tom 'i'. Fourth multiply each of the values in the cumulative normalized histogram by L-1, where L is the range of pixel values, normally 256. This step distributes the resulting pixel intensities across the full intensity range. Finally, transform each pixel in the original image by using our histogram as a look-up table. The resulting image will be contrast-enhanced.
The process of BBHE is the following. First, calculate the histogram of all pixel values. Second, calculate the mean based on the histogram. Third, create two histograms based on the mean value. One for all pixels with values lower than or equal to the mean and another for all pixels with values higher than the mean. Forth, normalize the two histograms separately and alter the normalized histogram to be cumulative. Finally, transform each pixel in the original image by using our histogram as a look-up table.
The process of DSIHE is similar to BBHE. We need to replace the mean value with the median value and repeat steps 3 to 5 of BBHE. 

### Results
There are five example images in the repo. The pictures with no suffixes are the original pictures. Histogram Equalization results end with -contrast. 
landscape.jpg and bridge.jpg: all three algorithms work well and yield similar results. 
flight.jpg: The histogram Equalization result is rather dark. This is because the algorithm has a "mean-shift" problem, where it shifts the mean intensity value to the middle gray level of the intensity range. BBHE adds more gray pixels while DSIHE adds more dark pixels to the image. 
couple.jpg: Histogram Equalization result is rather white this time due to the mean shift. BBHE and DSIHE enhance the image pretty well, which makes the dark original image brighter. 
hand.jpg: Once again, Histogram Equalization makes a dark image too white. BBHE makes the image look smoother while DSIHE adds too many gray pixels around the image.  

### Reflection
Histogram equalization is a widely used image enhancement algorithm in fields such as medical imaging. Learning the algorithm provided a good foundation for understanding non-deep image processing techniques. Furthermore, implementing the algorithm from this paper specifically exposed us to the shortcomings of the original algorithm, as well as how those shortcomings could be resolved. It was also interesting to see how the seemingly simple concept of histogram equalization can develop a group of effective algorithms. 

### Instructions for compiling and executing
The only required input is the path to the target image.

