/*
Garrett Partenza
September 10th, 2020
CS 7180 Advanced Perception
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

        // Read image    
        Mat img = imread(path, IMREAD_GRAYSCALE);
    
        // Calculate histogram
        int histSize = 256;
        float range[] = { 0, 256 };
        const float* histRange[] = { range };
        bool uniform = true;
        bool accumulate = false;
     
        Mat hist;
        calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, histRange, uniform, accumulate);
        
        // Normalize histogram
        Mat hist_normed;
        hist.convertTo(hist_normed, CV_32F, 1.0/img.total());
        
        // Convert normalized histogram to vector
        hist_normed = hist_normed.reshape(1, hist_normed.total()*hist_normed.channels());
        std::vector<float> vec = hist_normed.isContinuous()? hist_normed : hist_normed.clone();

        // Make normalized histogram cummulative
        float sum = 0;
        for(int i=0; i<vec.size(); i++){
            sum+=vec[i];
            vec[i]=sum;
        }
        
        // Create the conversion table   
        for(int i=0; i<vec.size(); i++){
            vec[i] = (histSize-1)*vec[i];
        }  

        // Apply conversion to original image
        for(int i=0; i<img.rows; i++){
            for(int j=0; j<img.cols; j++){              
                int pixel = img.at<uchar>(i,j);
                int equalized = vec[pixel];
                img.at<uchar>(i,j) = equalized;
            }
        }
         
        // Save image
        bool check = imwrite("result-image.jpg", img);
        
        if (check){
            cout << "Done" << endl;
        }else{
            cout << "Failed to save image" << endl;
        }

        return 0;
}
