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
using namespace cv;

void imshow(cuda::GpuMat*);
void imshow(cuda::GpuMat);
void imshow(Mat*);
void imshow(Mat);
void show_datatype(cuda::GpuMat*);
void show_datatype(cuda::GpuMat);
void show_datatype(Mat*);
void show_datatype(Mat);
void show_address(void*);
void contrast(cuda::GpuMat*, float);
void mask(Mat*, vector<vector<Point>>*);
void shrink(cuda::GpuMat*, float);
void rotate(Mat*, float, Scalar);
void rotate(Mat*, float);
bool crop(cuda::GpuMat*, Mat*, Mat*);
int leftEnd(Mat*);
int rightEnd(Mat*);
float hough(cuda::GpuMat*, int, int);
vector<Point> getMaxContour(Mat*);
vector<vector<Point>> denoise(vector<vector<Point>>*, int, int, float, float, float, float);

bool process(Mat &cpu, vector<Mat> *result_images) {
    cuda::GpuMat gpu;
    gpu.upload(cpu);
    cuda::cvtColor(gpu, gpu, COLOR_BGR2GRAY);
    cuda::normalize(gpu, gpu, 0, 255, NORM_MINMAX, -1);

    //allocate unified memory
    int height = gpu.rows;
    int width = gpu.cols;
    void *unified_ptr;    
    cudaMallocManaged(&unified_ptr, height * width);
    Mat shared_cpu(height, width, CV_8UC1, unified_ptr);
    cuda::GpuMat shared_gpu(height, width, CV_8UC1, unified_ptr);
    gpu.copyTo(shared_gpu);

	contrast(&shared_gpu, 200.f);

    int blockSize = shared_gpu.cols;
    if(blockSize%2 == 0)
        blockSize++;
    adaptiveThreshold(shared_cpu, shared_cpu, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, blockSize, 0.);
    
    vector<vector<Point>> plateContour(1, getMaxContour(&shared_cpu));
    //no contour found
    if(plateContour.size() == 0){
        printf("Detection failed!");
        return false;//exception
    }
    /*
    Scalar color(200);
    drawContours(shared_cpu, plateContour, 0, color, 5);
    imshow(shared_cpu);
    */

    mask(&shared_cpu, &plateContour);

    //shrink(&shared_gpu, 2.f);
    cuda::bitwise_not(shared_gpu, shared_gpu);    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(shared_cpu, contours, RETR_TREE, CHAIN_APPROX_NONE);
    if(contours.size() > 0){
        contours = denoise(&contours, shared_cpu.rows, shared_cpu.cols, 0.01f, 0.5f, 0.2f, 0.8f);
        cuda::bitwise_not(shared_gpu, shared_gpu);
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
	    rotate(&shared_cpu, angle, Scalar(255));
    }

	if( angle != 0.f ) rotate(&cpu, angle);

	cpu.copyTo((*result_images)[0]);

	bool isCrop = crop(&shared_gpu, &shared_cpu, &cpu);

	if(isCrop) cpu.copyTo((*result_images)[1]);
	else result_images->pop_back();

	cudaFree(unified_ptr);

    return true;
}

void imshow(cuda::GpuMat *frame){
    if(frame == NULL){
        printf("image is NULL\n");
        return;
    }
    Mat img;
    (*frame).download(img);
    imshow("gpu", img);
    waitKey(0);
}

void imshow(cuda::GpuMat frame){
    Mat img;
    frame.download(img);
    imshow("gpu", img);
    waitKey(0);
}

void imshow(Mat *frame){
    if(frame == NULL){
        printf("image is NULL\n");
        return;
    }
    imshow("cpu", *frame);
    waitKey(0);
}

void imshow(Mat frame){
    imshow("cpu", frame);
    waitKey(0);
}

void show_datatype(cuda::GpuMat *frame){
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

void show_datatype(cuda::GpuMat frame){
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

void show_datatype(Mat *frame){
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

void show_datatype(Mat frame){
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
    printf("address at %p\n", ptr);
}

//param contrast: - 減少對比度/+ 增加對比度
void contrast(cuda::GpuMat *gp, float contrast){
    float brightness = 0.f;
    float B = brightness / 255.0f;
    float c = contrast / 255.0f; 
    float k = tan((45.f + 44.f * c) / 180.f * 3.14159265f);

    //*src = (*src - 127.5 * (1.f - B)) * (k + 127.5f * (1.f + B))
    //type conversion is done with rounding and saturation
    (*gp).convertTo(*gp, (*gp).type(), 1., -(127.5 * (1.f - B)));
    (*gp).convertTo(*gp, (*gp).type(), k, 127.5f * (1.f + B)); 
}

vector<Point> getMaxContour(Mat* cp){
    vector<vector<Point> > contours;    
    findContours(*cp, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    double area;
    double maxArea = 0.;
    vector<Point> maxContour;
    for(int i = 0; i < contours.size(); i++){
        area = contourArea(contours[i]);
        if(area > maxArea){
            maxArea = area;
  	    	maxContour = contours[i];
        }
    }
    return maxContour;
}

//fill the outside of the contour with given color
void mask(Mat *cp, vector<vector<Point>> *contours){
    Mat mask((*cp).rows, (*cp).cols, CV_8UC1, 255);
    fillPoly(mask, *contours, 0);
    add((*cp), mask, (*cp));
}

void shrink(cuda::GpuMat *gp, float level){
    int h = (int)((*gp).cols*0.014*level);
    if(h == 0) h += 1;
    int w = (int)(0.5*h);
    if(w == 0) w += 1;
	//imshow(gp);
    Mat elem = getStructuringElement(MORPH_RECT, Size(w, h));
    Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter( MORPH_DILATE, CV_8UC1, elem);
    dilateFilter->apply((*gp), (*gp));
	//imshow(gp);
    elem = getStructuringElement(MORPH_RECT, Size(w, w));
    Ptr<cuda::Filter> erodeFilter = cuda::createMorphologyFilter( MORPH_ERODE, CV_8UC1, elem);
    erodeFilter->apply((*gp), (*gp));
	//imshow(gp);
}

vector<vector<Point>> denoise(vector<vector<Point>> *contours, int height, int width, float minWidth, float maxWidth, float minHeight, float maxHeight){
    vector<vector<Point>> number_contour;
    for(int i = 0; i < (*contours).size(); i++){
        Rect box = boundingRect((*contours)[i]);
        if(box.width > width * minWidth && box.width < width * maxWidth){
	    	if(box.height > height * minHeight && box.height < height * maxHeight)
				number_contour.push_back((*contours)[i]);
		}	
    }
    return number_contour;
}

int leftEnd(Mat *cp){
    for(int i = (int)((*cp).cols*0.1); i < (int)((*cp).cols*0.9); i += (int)((*cp).cols*0.01)){
        for(int j = (int)((*cp).rows*0.1); j < (int)((*cp).rows*0.9); j += (int)((*cp).rows*0.01)){
            if((*cp).at<char>(j,i) == 0){
                //circle((*cp), Point(i,j), 10, Scalar(150), 10);
				//imshow(cp);
                return i;
	    	}
		}
    }
    return (*cp).cols;
}

int rightEnd(Mat *cp){
    for(int i = (int)((*cp).cols*0.9); i > (int)((*cp).cols*0.1); i -= (int)((*cp).cols*0.01)){
        for(int j = (int)((*cp).rows*0.1); j < (int)((*cp).rows*0.9); j += (int)((*cp).rows*0.01)){
            if((*cp).at<char>(j,i) == 0){
                //circle((*cp), Point(i,j), 10, Scalar(150), 10);
				//imshow(cp);
                return i;
	    	}
		}
    }
    return 0;
}

float hough(cuda::GpuMat *gp, int minLineLength,int maxLineGap){
    cuda::GpuMat tmp;
    cuda::resize(*gp, tmp, Size((*gp).cols, (int)((*gp).cols * 0.15)), 0., 0., INTER_LINEAR);
    cuda::bitwise_not(tmp, tmp);
    Ptr< cuda::HoughSegmentDetector> hough = cuda::createHoughSegmentDetector(1.f, (float)(CV_PI / 180.f), minLineLength, maxLineGap, 40);
    cuda::GpuMat lines_gpu;
    hough->detect(tmp, lines_gpu);

    vector<Vec4i> lines;
    if (!lines_gpu.empty()){
        lines.resize(lines_gpu.cols);
        Mat h_lines(1, lines_gpu.cols, CV_32SC4, &lines[0]);
        lines_gpu.download(h_lines);
    }
    else
		return 0.;
    
    if(lines.size() > 6){
		/*
		//for drawing lines        
	    Mat draw;
	    (tmp).download(draw);
		*/
        int size = (lines.size() < 20) ? lines.size() : 20;
		double radians = 0;
        for(int i = 0; i < size; i++){
            Vec4i line = lines[i];
            radians += atan2(line[3] - line[1], line[2] - line[0]);            
            //line(draw, Point(line[0], line[1]), Point( line[2], line[3]), Scalar(200), 8);
        }
        //imshow(draw);
		double ratio = atan( ( (double)((*gp).rows) / (double)((*gp).cols * 0.15) ) * tan(radians) ) / radians;
        double angle = radians * ratio * 57.3 / (double)lines.size();//57.3 = 180/3.14
        return (2. < angle || angle < -2.) ? angle : 0.f;
    }
    return 0.f;
}

void rotate(Mat *cp, float angle, Scalar fill){
    int height = (*cp).rows;
    int width = (*cp).cols;
    Point2f image_center = Point2f(width*0.5, height*0.5);
    Mat rotation_mat = getRotationMatrix2D(image_center, angle, 1.);
    warpAffine(*cp, *cp, rotation_mat, (*cp).size(), INTER_LINEAR, BORDER_CONSTANT, fill);
    //cuda::warpAffine(*gp, *gp, rotation_mat, Size(bound_w, bound_h), INTER_LINEAR, BORDER_CONSTANT, fill);
}

void rotate(Mat *cp, float angle){
    int height = (*cp).rows;
    int width = (*cp).cols;
    Point2f image_center = Point2f(width*0.5, height*0.5);
    Mat rotation_mat = getRotationMatrix2D(image_center, angle, 1.);
    warpAffine(*cp, *cp, rotation_mat, (*cp).size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255,255,255));
    //cuda::warpAffine(*gp, *gp, rotation_mat, Size(bound_w, bound_h), INTER_LINEAR, BORDER_CONSTANT, fill);
}

bool crop(cuda::GpuMat *gp, Mat *cp, Mat *color){
	cuda::bitwise_not(*gp, *gp);
	int w = gp->cols;
    int h = (int)(gp->rows*0.1);
	if(h == 0) h += 1;
	Mat elem = getStructuringElement(MORPH_RECT, Size(w, h));
	/*
    Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, CV_8UC1, elem);  	
	dilateFilter->apply(*gp, *gp);
	*/
	dilate(*cp, *cp, elem);
	//imshow(gp);

	vector<Point> contour = getMaxContour(cp);
	if(contour.size() > 0){
		vector<Point> contours_poly( contour.size() );
		approxPolyDP( contour, contours_poly, int(contour.size()*0.2), true );
        Rect ROI = boundingRect( contours_poly ) & Rect(0, 0, color->cols, color->rows);
		*color = (*color)(ROI);
		return true;
	}
    else return false;
}
