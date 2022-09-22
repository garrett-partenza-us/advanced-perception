/*
Jamie Sun
September 19th, 2022
CS 7180 Advanced Perception
This file is an implementation of Brightness Preserving Bi-Histogram Equalization(BBHE). 
*/

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
        string path; 
        if (argc > 1){
            path = argv[1];
        }
        
        // Read image.    
        Mat img = imread(path, IMREAD_GRAYSCALE);

        // Calculate histogram.
        int histSize = 256;
        float range[] = { 0, 256 };
        const float* histRange[] = { range };
        bool uniform = true;
        bool accumulate = false;
     
        Mat hist;
        calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, histRange, uniform, accumulate);
        printf("Calculate histogram. The size of hist is %zu.\n", hist.total());

        float sumOfPixels = 0;
        // Calculate mean.
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                sumOfPixels += float(img.at<uchar>(i, j));
            }
        }
        int mean = sumOfPixels / img.total();
        printf("sum is %f, total is %zu.\n", sumOfPixels, img.total());
        printf("sum is %f, total is %d.\n", sumOfPixels, img.cols * img.rows);
        printf("Calculate mean, mean is %d. \n", mean);
        
        float checkHistTotal = 0;
        float checkNewHistTotal = 0;
        //Split the histogram into lower than or equal to mean and higher than mean. 
        Mat histL = Mat::zeros(1, mean + 1, CV_32F);
        for (int i = 0; i <= mean; i++) {
            histL.at<float>(i) = hist.at<float>(i);
        }

        Mat histH = Mat::zeros(1, 256 - mean - 1, CV_32F);
        for (int i = 0; i <= 256 - mean - 1; i++) {
            int histIndex = i + mean + 1;
            histH.at<float>(i) = hist.at<float>(histIndex);
        }

        for (int i = 0; i < histL.cols; i++) {
            checkNewHistTotal += histL.at<float>(i);
        }
        printf("checkNewHistTotal is %f\n", checkNewHistTotal);

        for (int i = 0; i < histH.cols; i++) {
            checkNewHistTotal += histH.at<float>(i);
        }
        printf("checkNewHistTotal after adding high is %f\n", checkNewHistTotal);

        for (int i = 0; i < 256; i++) {
            checkHistTotal += hist.at<float>(i);
        }
        printf("checkHistTotal is %f\n", checkHistTotal);


        printf("Split the histogram\n");

        // Normalize two histograms seperately.
        Mat histL_normed;
        Mat histH_normed;
        double totalValueL = 0;
        double totalValueH = 0;
        for (int i = 0; i < histL.cols; i++) {
            totalValueL += histL.at<float>(i);
        }
        for (int i = 0; i < histH.cols; i++) {
            totalValueH += histH.at<float>(i);
        }
        printf("Lower part total is %f\n", totalValueL);
        printf("Higher part total is %f\n", totalValueH);
        histL.convertTo(histL_normed, CV_32F, 1.0 / totalValueL);
        histH.convertTo(histH_normed, CV_32F, 1.0 / totalValueH);
        printf("Lower part total is %zu, channels are %d\n", histL_normed.total(),histL_normed.channels());
        printf("Higher part total is %zu, channels are %d\n", histH_normed.total(),histH_normed.channels());
        
        // Convert normalized histogram to vector.
        Mat hist_normed;
        hconcat(histL_normed, histH_normed, hist_normed);
        printf("Before reshape: %d, %d.\n", hist_normed.cols, hist_normed.rows);
        hist_normed = hist_normed.reshape(1, hist_normed.total()*hist_normed.channels());
        printf("After reshape: %d, %d.\n", hist_normed.cols, hist_normed.rows);
        std::vector<float> vec = hist_normed.isContinuous()? hist_normed : hist_normed.clone();

        // Make normalized histogram cumulative
        float sumL = 0;
        for(int i = 0; i <= mean; i++){
            sumL += vec[i];
            vec[i] = sumL;
        }
        
        float sumH = 0;
        for(int i = mean+1; i <= 255; i++){
            sumH += vec[i];
            vec[i] = sumH;
        }
        
        // Create the conversion table 
        for(int i = 0; i <= mean; i++){
            vec[i] = mean * vec[i];
        }
        
        for(int i = mean+1; i <= 255; i++){
            vec[i] = mean + 1 + (255 - mean - 1) * vec[i];
        }

        printf("The size of vec is %lu\n", vec.size());

        // Apply conversion to original image
        for(int i=0; i<img.rows; i++){
            for(int j=0; j<img.cols; j++){              
                int pixel = img.at<uchar>(i,j);
                int equalized = vec[pixel];
                img.at<uchar>(i,j) = equalized;
            }
        }
         
        // Save image
        bool check = imwrite("BBHE.jpg", img);
        
        if (check){
            cout << "Done" << endl;
        }else{
            cout << "Failed to save image" << endl;
        }

        return 0;
}
