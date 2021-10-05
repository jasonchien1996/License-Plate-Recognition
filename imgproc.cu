/*
nvcc imgproc.cu -o libimgproc.so --shared --compiler-options '-fPIC' -I/usr/local/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_cudaimgproc -lopencv_highgui -lopencv_cudaarithm -lopencv_cudawarping -lopencv_cudafilters -gencode=arch=compute_72,code=sm_72
*/

#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include "imgproc.cuh"
using namespace std;

void imshow(cv::cuda::GpuMat*);
void imshow(cv::cuda::GpuMat);
void imshow(cv::Mat*);
void imshow(cv::Mat);
void show_datatype(cv::cuda::GpuMat*);
void show_datatype(cv::cuda::GpuMat);
void show_datatype(cv::Mat*);
void show_datatype(cv::Mat);
void show_address(void*);
void contrast(cv::cuda::GpuMat*, float);
void mask(cv::Mat*, std::vector<std::vector<cv::Point>>*);
void shrink(cv::cuda::GpuMat*, float);
void rotate(cv::Mat*, float, cv::Scalar);
void rotate(cv::Mat*, float);
bool crop(cv::cuda::GpuMat*, cv::Mat*, cv::Mat*);
int leftEnd(cv::Mat*);
int rightEnd(cv::Mat*);
float hough(cv::cuda::GpuMat*, int, int);
vector<cv::Point> getMaxContour(cv::Mat*);
vector<vector<cv::Point>> denoise(vector<vector<cv::Point>>*, int, int, float, float, float, float);

bool process(cv::Mat &cpu, vector<cv::Mat> *result_images) {
    cv::cuda::GpuMat gpu;
    gpu.upload(cpu);
    cv::cuda::cvtColor(gpu, gpu, cv::COLOR_BGR2GRAY);
    cv::cuda::normalize(gpu, gpu, 0, 255, cv::NORM_MINMAX, -1);

    //allocate unified memory
    int height = gpu.rows;
    int width = gpu.cols;
    void *unified_ptr;    
    cudaMallocManaged(&unified_ptr, height * width);
    cv::Mat shared_cpu(height, width, CV_8UC1, unified_ptr);
    cv::cuda::GpuMat shared_gpu(height, width, CV_8UC1, unified_ptr);
    gpu.copyTo(shared_gpu);

	contrast(&shared_gpu, 200.f);

    int blockSize = shared_gpu.cols;
    if(blockSize%2 == 0)
        blockSize++;
    cv::adaptiveThreshold(shared_cpu, shared_cpu, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, 0.);
    
    vector<vector<cv::Point>> plateContour(1, getMaxContour(&shared_cpu));
    //no contour found
    if(plateContour.size() == 0){
        printf("Detection failed!");
        return false;//exception
    }
    /*
    cv::Scalar color(200);
    cv::drawContours(shared_cpu, plateContour, 0, color, 5);
    imshow(shared_cpu);
    */

    mask(&shared_cpu, &plateContour);

    //shrink(&shared_gpu, 2.f);
    cv::cuda::bitwise_not(shared_gpu, shared_gpu);    
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    cv::findContours(shared_cpu, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
    if(contours.size() > 0){
        contours = denoise(&contours, shared_cpu.rows, shared_cpu.cols, 0.01f, 0.5f, 0.2f, 0.8f);
        cv::cuda::bitwise_not(shared_gpu, shared_gpu);
        mask(&shared_cpu, &contours);
    }

    int l = leftEnd(&shared_cpu);
    int r = rightEnd(&shared_cpu);

    float angle = 0.f;
    if( r > l ){
		int temp = r - l;
		shrink(&shared_gpu, temp * 0.004f);
        int minLineLength = (int)(temp*0.4);
        int maxLineGap = (int)(shared_cpu.cols*0.2);
        angle = hough(&shared_gpu, minLineLength, maxLineGap);
	    rotate(&shared_cpu, angle, cv::Scalar(255));
    }

	if( angle != 0.f ) rotate(&cpu, angle);

	cpu.copyTo((*result_images)[0]);

	bool isCrop = crop(&shared_gpu, &shared_cpu, &cpu);

	if(isCrop) cpu.copyTo((*result_images)[1]);
	else result_images->pop_back();

	cudaFree(unified_ptr);

    return true;
}

void imshow(cv::cuda::GpuMat *frame){
    if(frame == NULL){
        printf("image is NULL\n");
        return;
    }
    cv::Mat img;
    (*frame).download(img);
    cv::imshow("gpu", img);
    cv::waitKey(0);
}

void imshow(cv::cuda::GpuMat frame){
    cv::Mat img;
    frame.download(img);
    cv::imshow("gpu", img);
    cv::waitKey(0);
}

void imshow(cv::Mat *frame){
    if(frame == NULL){
        printf("image is NULL\n");
        return;
    }
    cv::imshow("cpu", *frame);
    cv::waitKey(0);
}

void imshow(cv::Mat frame){
    cv::imshow("cpu", frame);
    cv::waitKey(0);
}

void show_datatype(cv::cuda::GpuMat *frame){
    switch((*frame).type()){
        case 0:
	    printf("data type %s\n", "CV_8UC1");
            break;
        case 16:
	    printf("data type %s\n", "CV_8UC3");
            break;
		default:
            printf("data type %d\n", (*frame).type());            
    }    
}

void show_datatype(cv::cuda::GpuMat frame){
    switch(frame.type()){
        case 0:
	    printf("data type %s\n", "CV_8UC1");
            break;
        case 16:
	    printf("data type %s\n", "CV_8UC3");
            break;
		default:
            printf("data type %d\n", frame.type());            
    }    
}

void show_datatype(cv::Mat *frame){
    switch((*frame).type()){
        case 0:
	    printf("data type %s\n", "CV_8UC1");
            break;
        case 16:
	    printf("data type %s\n", "CV_8UC3");
            break;
		default:
            printf("data type %d\n", (*frame).type());            
    }    
}

void show_datatype(cv::Mat frame){
    switch(frame.type()){
        case 0:
	    printf("data type %s\n", "CV_8UC1");
            break;
        case 16:
	    printf("data type %s\n", "CV_8UC3");
            break;
		default:
            printf("data type %d\n", frame.type());            
    }    
}

void show_address(void *ptr){
    std::printf("address at %p\n", ptr);
}

//param contrast: - 減少對比度/+ 增加對比度
void contrast(cv::cuda::GpuMat *gp, float contrast){
    float brightness = 0.f;
    float B = brightness / 255.0f;
    float c = contrast / 255.0f; 
    float k = tan((45.f + 44.f * c) / 180.f * 3.14159265f);

    //*src = (*src - 127.5 * (1.f - B)) * (k + 127.5f * (1.f + B))
    //type conversion is done with rounding and saturation
    (*gp).convertTo(*gp, (*gp).type(), 1., -(127.5 * (1.f - B)));
    (*gp).convertTo(*gp, (*gp).type(), k, 127.5f * (1.f + B)); 
}

vector<cv::Point> getMaxContour(cv::Mat* cp){
    vector<vector<cv::Point> > contours;    
    cv::findContours(*cp, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    
    double area;
    double maxArea = 0.;
    vector<cv::Point> maxContour;
    for(int i = 0; i < contours.size(); i++){
        area = cv::contourArea(contours[i]);
        if(area > maxArea){
            maxArea = area;
  	    	maxContour = contours[i];
        }
    }
    return maxContour;
}

//fill the outside of the contour with given color
void mask(cv::Mat *cp, std::vector<std::vector<cv::Point>> *contours){
    cv::Mat mask((*cp).rows, (*cp).cols, CV_8UC1, 255);
    cv::fillPoly(mask, *contours, 0);
    cv::add((*cp), mask, (*cp));
}

void shrink(cv::cuda::GpuMat *gp, float level){
    int h = (int)((*gp).cols*0.014*level);
    if(h == 0) h += 1;
    int w = (int)(0.5*h);
    if(w == 0) w += 1;
	//imshow(gp);
    cv::Mat elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(w, h));
    cv::Ptr<cv::cuda::Filter> dilateFilter = cv::cuda::createMorphologyFilter( cv::MORPH_DILATE, CV_8UC1, elem);
    dilateFilter->apply((*gp), (*gp));
	//imshow(gp);
    elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(w, w));
    cv::Ptr<cv::cuda::Filter> erodeFilter = cv::cuda::createMorphologyFilter( cv::MORPH_ERODE, CV_8UC1, elem);
    erodeFilter->apply((*gp), (*gp));
	//imshow(gp);
}

vector<vector<cv::Point>> denoise(vector<vector<cv::Point>> *contours, int height, int width, float minWidth, float maxWidth, float minHeight, float maxHeight){
    std::vector<std::vector<cv::Point>> number_contour;
    for(int i = 0; i < (*contours).size(); i++){
        cv::Rect box = cv::boundingRect((*contours)[i]);
        if(box.width > width * minWidth && box.width < width * maxWidth){
	    	if(box.height > height * minHeight && box.height < height * maxHeight)
				number_contour.push_back((*contours)[i]);
		}	
    }
    return number_contour;
}

int leftEnd(cv::Mat *cp){
    for(int i = (int)((*cp).cols*0.1); i < (int)((*cp).cols*0.9); i += (int)((*cp).cols*0.01)){
        for(int j = (int)((*cp).rows*0.1); j < (int)((*cp).rows*0.9); j += (int)((*cp).rows*0.01)){
            if((*cp).at<char>(j,i) == 0){
                //cv::circle((*cp), cv::Point(i,j), 10, cv::Scalar(150), 10);
		//imshow(cp);
                return i;
	    	}
		}
    }
    return (*cp).cols;
}

int rightEnd(cv::Mat *cp){
    for(int i = (int)((*cp).cols*0.9); i > (int)((*cp).cols*0.1); i -= (int)((*cp).cols*0.01)){
        for(int j = (int)((*cp).rows*0.1); j < (int)((*cp).rows*0.9); j += (int)((*cp).rows*0.01)){
            if((*cp).at<char>(j,i) == 0){
                //cv::circle((*cp), cv::Point(i,j), 10, cv::Scalar(150), 10);
		//imshow(cp);
                return i;
	    	}
		}
    }
    return 0;
}

float hough(cv::cuda::GpuMat *gp, int minLineLength,int maxLineGap){
    cv::cuda::GpuMat tmp;
    cv::cuda::resize(*gp, tmp, cv::Size((*gp).cols, (int)((*gp).cols * 0.15)), 0., 0., cv::INTER_LINEAR);
    cv::cuda::bitwise_not(tmp, tmp);
    cv::Ptr< cv::cuda::HoughSegmentDetector> hough = cv::cuda::createHoughSegmentDetector(1.f, (float)(CV_PI / 180.f), minLineLength, maxLineGap, 40);
    cv::cuda::GpuMat lines_gpu;
    hough->detect(tmp, lines_gpu);

    std::vector<cv::Vec4i> lines;
    if (!lines_gpu.empty()){
        lines.resize(lines_gpu.cols);
        cv::Mat h_lines(1, lines_gpu.cols, CV_32SC4, &lines[0]);
        lines_gpu.download(h_lines);
    }
    else
		return 0.;
    
    if(lines.size() > 6){
		/*
		//for drawing lines        
	    cv::Mat draw;
	    (tmp).download(draw);
		*/
        int size = (lines.size() < 20) ? lines.size() : 20;
		double radians = 0;
        for(int i = 0; i < size; i++){
            cv::Vec4i line = lines[i];
            radians += atan2(line[3] - line[1], line[2] - line[0]);            
            //cv::line(draw, cv::Point(line[0], line[1]), cv::Point( line[2], line[3]), cv::Scalar(200), 8);
        }
        //imshow(draw);
		double ratio = atan( ( (double)((*gp).rows) / (double)((*gp).cols * 0.15) ) * tan(radians) ) / radians;
        double angle = radians * ratio * 57.3 / (double)lines.size();//57.3 = 180/3.14
        return (2. < angle || angle < -2.) ? angle : 0.f;
    }
    return 0.f;
}

void rotate(cv::Mat *cp, float angle, cv::Scalar fill){
    int height = (*cp).rows;
    int width = (*cp).cols;
    cv::Point2f image_center = cv::Point2f(width*0.5, height*0.5);
    cv::Mat rotation_mat = cv::getRotationMatrix2D(image_center, angle, 1.);
    cv::warpAffine(*cp, *cp, rotation_mat, (*cp).size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, fill);
    //cv::cuda::warpAffine(*gp, *gp, rotation_mat, cv::Size(bound_w, bound_h), cv::INTER_LINEAR, cv::BORDER_CONSTANT, fill);
}

void rotate(cv::Mat *cp, float angle){
    int height = (*cp).rows;
    int width = (*cp).cols;
    cv::Point2f image_center = cv::Point2f(width*0.5, height*0.5);
    cv::Mat rotation_mat = cv::getRotationMatrix2D(image_center, angle, 1.);
    cv::warpAffine(*cp, *cp, rotation_mat, (*cp).size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255,255,255));
    //cv::cuda::warpAffine(*gp, *gp, rotation_mat, cv::Size(bound_w, bound_h), cv::INTER_LINEAR, cv::BORDER_CONSTANT, fill);
}

bool crop(cv::cuda::GpuMat *gp, cv::Mat *cp, cv::Mat *color){
	cv::cuda::bitwise_not(*gp, *gp);
	int w = gp->cols;
    int h = (int)(gp->rows*0.1);
	if(h == 0) h += 1;
	cv::Mat elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(w, h));
	/*
    cv::Ptr<cv::cuda::Filter> dilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, elem);  	
	dilateFilter->apply(*gp, *gp);
	*/
	cv::dilate(*cp, *cp, elem);
	//imshow(gp);

	vector<cv::Point> contour = getMaxContour(cp);
	if(contour.size() > 0){
		vector<cv::Point> contours_poly( contour.size() );
		approxPolyDP( contour, contours_poly, int(contour.size()*0.2), true );
        cv::Rect ROI = boundingRect( contours_poly ) & cv::Rect(0, 0, color->cols, color->rows);
		*color = (*color)(ROI);
		return true;
	}
    else return false;
}
