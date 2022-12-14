Garrett Partenza and Jiameng Sung
CS 7180 Advanced Computer Vision
December 14th, 2022

# SuperTiny Resolution: Designing a satellite super resolution model in the Tinygrad machine learning framework

This repository contains our submission for the final project, multiple super-resolution implementations in the tinygrad machine learning framework.

To exectute, log into the discovery cluster and provision a GPU (we used the A100). Create a virtual environment (we used pipenv) and install the python library requirements outlined in the requirements.txt file. Then, run train.py with the OS environment flag GPU=1 for GPU training. You can swap out models on line 26 of train.py for kernel, srcnn, and espcnn. 


