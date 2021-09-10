/*
nvcc display.cpp imgproc.cu -o display.out -I/usr/local/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_cudaimgproc -lopencv_highgui -lopencv_cudaarithm -lopencv_cudawarping -lopencv_cudafilters -gencode=arch=compute_72,code=sm_72
*/

#include "imgproc.cuh"
#include <opencv2/opencv.hpp>
using namespace std;

int main() {
	cv::Mat img = cv::imread("./picture/2.jpg");
	cv::Mat a,b,c;
	vector<cv::Mat> images = {a, b, c};
    if(process(img, &images)){
		for(cv::Mat mat:images){
			cv::imshow("mat", mat);
			cv::waitKey(0);
		}
	}	
	return 0;
}
