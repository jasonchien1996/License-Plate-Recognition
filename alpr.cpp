#include "car.hpp"
#include "util.hpp"
#include "imgproc.cuh"
#include "darknet.h"
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

//for onnxruntime
#define useCUDA true
#define instanceName "plate-detector-inference"
#define modelFilepath "./checkpoints/custom-416.onnx"
#define iou_threshold .45f
#define conf_threshold .5f

//for darknet
#define OCR_TRESHOLD .3f

#define VIDEO_PATH "./video/video (3).mp4"
#define CONSEC_CORRECT 6
#define MAX_IDLE 6

using namespace cv;
using namespace std;
using namespace util;
using namespace std::chrono;

char OCR_WEIGHTS[] = "ocr/ocr-net.weights";
char OCR_NETCFG[]  = "ocr/ocr-net.cfg";
char OCR_DATASET[] = "ocr/ocr-net.data";

template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v);
ostream& operator<<(ostream&, const ONNXTensorElementDataType&);
bool stop(int, string&);

int main() {
	VideoCapture capture( VIDEO_PATH );
	if( !capture.isOpened() ) {
		cout << "Unable to open file!";
		return 0;
	}		
	
	//initialize onnxruntime
	cout << "initialize onnxruntime\n"; 
	if (useCUDA) cout << "Inference Execution Provider: CUDA\n";
	else cout << "Inference Execution Provider: CPU\n";
	
	Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName);
	Ort::SessionOptions sessionOptions;
	//sessionOptions.SetIntraOpNumThreads(1);
	
	if (useCUDA) {
		// Using CUDA backend
		// https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L329
		OrtCUDAProviderOptions cuda_options;
		cuda_options.device_id = 0;
		sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }
	
	// Sets graph optimization level
	// Available levels are
	// ORT_DISABLE_ALL -> To disable all optimizations
	// ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
	// ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
	// ORT_ENABLE_ALL -> To Enable All possible optimizations
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	Ort::Session session(env, modelFilepath, sessionOptions);

	Ort::AllocatorWithDefaultOptions allocator;

	size_t numInputNodes = session.GetInputCount();
	size_t numOutputNodes = session.GetOutputCount();

	cout << "Number of Input Nodes: " << numInputNodes << endl;
	cout << "Number of Output Nodes: " << numOutputNodes << endl;

	const char* inputName = session.GetInputName(0, allocator);
	cout << "Input Name: " << inputName << endl;

	Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

	ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
	cout << "Input Type: " << inputType << endl;

	vector<int64_t> inputDims = inputTensorInfo.GetShape();
	cout << "Input Dimensions: " << inputDims << endl;

	const char* outputName = session.GetOutputName(0, allocator);
	cout << "Output Name: " << outputName << endl;

	Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
	auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

	ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
	cout << "Output Type: " << outputType << endl;

	vector<int64_t> outputDims = outputTensorInfo.GetShape();
	cout << "Output Dimensions: " << outputDims << endl << endl;

	//initialize darknet
	cout << "initialize darknet\n"; 
	network *ocr_net = load_network( OCR_NETCFG, OCR_WEIGHTS, 0 );
	metadata ocr_meta = get_metadata( OCR_DATASET );
	set_batch_network(ocr_net, 1);
	
	int imgCount = 0;           //number of images of the current vehicle
    int vehicleCount = 1;       //number of vehicles
    int idle = 0;               //consecutive images with no plate
    bool skip = false;          //set to true if early stoppping

	string date = getDate();
	if(opendir(date.c_str()) == nullptr) {
		cout << "create directory " << date << endl;
		mkdir(date.c_str(), 0775);
	}
	fstream record;
    record.open( "./" + date + "/" + date + ".txt", ios::out);
	string crop_dir_path = "./" + date + "/" + to_string(vehicleCount);			

	Car plate;
	vector<Mat> images;
	while(1) {
		auto t1 = high_resolution_clock::now();
		
		auto now = getDate().c_str();
		if( now != date ) {
			record.close();
            date = now;            
			if(opendir(date.c_str()) == nullptr) {
				cout << "create directory " << date << endl;
				mkdir(date.c_str(), 0775);
			}
            record.open( "./" + date + "/" + date + ".txt", ios::out);
		}
		crop_dir_path = "./" + date + "/" + to_string(vehicleCount);

		Mat imageBGR, resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
		if( !capture.read(imageBGR) ) {
			cout << "video ended or error\n";			
			continue;
		}
		//imageBGR = imread("./picture/1.jpg");
		
		resize(imageBGR, resizedImageBGR, Size(416, 416), InterpolationFlags::INTER_CUBIC);
		resizedImageBGR.convertTo(resizedImage, CV_32F, 1.0 / 255);
		dnn::blobFromImage(resizedImage, preprocessedImage);

		size_t inputTensorSize = 519168;//416*416*3=519168
		vector<float> temp(inputTensorSize);
		temp.assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>());

		vector<float> inputTensorValues(inputTensorSize);		
		for(int i = 0; i < 173056; ++i){		
			inputTensorValues[i*3] = temp[i];
			inputTensorValues[i*3+1] = temp[i+173056];
			inputTensorValues[i*3+2] = temp[i+346112];
		}

		vector<const char*> inputNames{inputName};
		vector<const char*> outputNames{outputName};
		vector<Ort::Value> inputTensors;
		inputDims[0] = 1;//batch size
		Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
		inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(), inputDims.size()));
		float* floatarrinput = inputTensors[0].GetTensorMutableData<float>();

		auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1, outputNames.data(), 1);
		auto info = outputTensors[0].GetTensorTypeAndShapeInfo();	
		//cout << "output tensor shape " << info.GetShape() << endl;

		int num_boxes = int(info.GetElementCount()*0.2);
		float* floatarr = outputTensors[0].GetTensorMutableData<float>();
		vector<util::box> boxes;
		for(int i = 0; i < num_boxes; ++i) {
			int row = i*5;
			if( floatarr[row+4] < conf_threshold) continue;
			util::box *b = create_box(floatarr[row], floatarr[row+1], floatarr[row+2], floatarr[row+3], floatarr[row+4]);
			boxes.push_back(*b);
			delete b; 
		}
		
		if(boxes.size() == 0) {
			idle += 1;
			if( idle == MAX_IDLE ) {
				if( skip == false ) {
                    record << crop_dir_path << endl;
                    record << "number of images scanned: " << imgCount << endl;
                    record << "predicted category: " << plate.getDominantCategory() << endl;
                    record << "predicted result: " << plate.vote() << endl << endl;
				}
				skip = false;
				imgCount = 0;
				vehicleCount += 1;
			}

			cout << "idle\n";
			auto t2 = high_resolution_clock::now();
			auto duration = duration_cast<microseconds>(t2 - t1);
			printf("%4.3fseconds\n", duration.count()*0.000001);
			cout << "=================================="<< endl;
			continue;
		}

		nms(boxes, iou_threshold);
		
		if(boxes.size() > 1) {
			cout << "too many plates found" << endl;
			auto t2 = high_resolution_clock::now();
			auto duration = duration_cast<microseconds>(t2 - t1);
			printf("%4.3fseconds\n", duration.count()*0.000001);
			cout << "=================================="<< endl;
			continue;
		}
		
		if( skip ) {
			cout << "skip\n";
			continue;
		}
		imgCount += 1;
		idle = 0;

		util::box &bb = boxes[0];
		int x = int(bb.tl[0]*imageBGR.cols);
		int y = int(bb.tl[1]*imageBGR.rows);
		int width = int((bb.br[0]-bb.tl[0])*imageBGR.cols);
		int height = int((bb.br[1]-bb.tl[1])*imageBGR.rows);
		
		if( x + width + 5 <= imageBGR.cols) width += 5;
		if( y + height + 5 <= imageBGR.rows) height += 5;
		if( x - 5 > 0) {
			x -= 5;
			width += 5;
		}
		if( y - 5 > 0) {
			y -= 5;
			height += 5;
		}

		Rect myROI(x, y, width, height);
		Mat croppedBGR = imageBGR(myROI);
		if(croppedBGR.cols > 416) resize(croppedBGR, croppedBGR, Size(416, 208), InterpolationFlags::INTER_CUBIC);
		//imshow("croppedBGR", croppedBGR);
		//waitKey(0);
		
		if( imgCount%50 == 1){
			auto path = crop_dir_path.c_str();
			auto dest = crop_dir_path + "/" + to_string(imgCount) + ".png";
			if(opendir(path) == nullptr) {
				cout << "create directory " << path << endl;
				mkdir(path, 0775);		
			}
			cv::imwrite(dest, croppedBGR);
		}

		Mat a,b;//create dummy Mat for space
		images.push_back(a);
		images.push_back(b);
		static int num_frame = 0;
		num_frame += 1;
		if( process(croppedBGR, &images) ) {
			for( cv::Mat &img : images) {		
				//cv::imshow("img", img);
				//cv::waitKey(0);
				if( skip ) {
					cout << "skip\n";
					break;
				}

				image im = mat_to_image(img);
				
				string OCRresult;
				int conf;
				ocr( ocr_net, ocr_meta, im, OCR_TRESHOLD , OCRresult, conf);
				
				if( OCRresult.length() < 6 || OCRresult.length() > 7 ) {
					cout << "ocr: " << OCRresult << endl;
					cout << "category: " << endl;
					cout << "confidence: " << endl;
					continue;
				}
				int category = -1;

				bool imply_category1 = plate.recognize( OCRresult, category );

				if ( OCRresult == "") {
					cout << "ocr: " << endl;
					cout << "category: " << endl;
					cout << "confidence: " << endl;
					continue;
				}
				cout << "ocr: " << OCRresult << endl;
				cout << "category: " << category << endl;
				cout << "confidence: " << conf << endl;
				
				plate.setCounter( OCRresult, category, conf, imply_category1 );

				if( stop(category, OCRresult) ) {					
					record << crop_dir_path << endl;
					record << "" << endl;
                    record << "number of images scanned: " << imgCount << endl;
                    record << "predicted category: " << category << endl;
                    record << "predicted result: " << OCRresult << endl << endl;
					skip = true;	
								
					//return 0;
				}
			}
			auto t2 = high_resolution_clock::now();
			auto duration = duration_cast<microseconds>(t2 - t1);
			printf("%4.3fseconds\n", duration.count()*0.000001);
			cout << "=================================="<< endl;		
		}
		images.clear();
	} 	
	capture.release();    
	return 0;
}

bool stop(int category, string &result) {
	bool Stop = true;
	static array<char, 7> currentSeven = {'#', '#', '#', '#', '#', '#', '#'};
	static array<char, 6> currentSix = {'#', '#', '#', '#', '#', '#'};
	static array<unsigned, 7> consec7 = {0, 0, 0, 0, 0, 0, 0};
	static array<unsigned, 6> consec6 = {0, 0, 0, 0, 0, 0};

	switch(category) {
		case 0:
			for( int i = 0; i < 7 ; ++i ) cout << currentSeven[i];
			cout << endl;
			for( int i = 0; i < 7 ; ++i ) cout << consec7[i];
			cout << endl;

			for( int i = 0; i < 7 ; ++i ) {
				if( result[i] == '#' ) continue;
				if( consec7[i] < CONSEC_CORRECT ) {
					if( currentSeven[i] == '#') {
						currentSeven[i] = result[i];
						consec7[i] = 1;
					}
					else if( currentSeven[i] == result[i] )
						consec7[i] += 1;
					else {
						currentSeven[i] = result[i];
						consec7[i] = 1;
					}
					if( consec7[i] < CONSEC_CORRECT ) Stop = false;
				}
				result[i] = currentSeven[i];
			}

			for( int i = 0; i < 7 ; ++i ) cout << currentSeven[i];
			cout << endl;
			for( int i = 0; i < 7 ; ++i ) cout << consec7[i];
			cout << endl;
			cout << "----------------------------------" << endl;

			if(Stop) {
				cout << "predicted result: \n";
				cout << "----------------------------------" << endl;
				for( int i = 0; i < 7 ; ++i ) cout << currentSeven[i];
				cout << endl;
			}
			break;
		case 1:
		case 2:
			for( int i = 0; i < 6 ; ++i ) cout << currentSix[i];
			cout << endl;
			for( int i = 0; i < 6 ; ++i ) cout << consec6[i];
			cout << endl;

			for( int i = 0; i < 6 ; ++i ) {
				if( result[i] == '#' ) continue;
				if( consec6[i] < CONSEC_CORRECT ) {
					if( currentSix[i] == '#') {
						currentSix[i] = result[i];
						consec6[i] = 1;
					}
					else if( currentSix[i] == result[i] )
						consec6[i] += 1;
					else {
						currentSix[i] = result[i];
						consec6[i] = 1;
					}
					if( consec6[i] < CONSEC_CORRECT ) Stop = false;
				}
				result[i] = currentSix[i];
			}

			for( int i = 0; i < 6 ; ++i ) cout << currentSix[i];
			cout << endl;
			for( int i = 0; i < 6 ; ++i ) cout << consec6[i];
			cout << endl;
			cout << "----------------------------------" << endl;

			if(Stop) {
				cout << "predicted category: " << category << endl;
				cout << "predicted result: \n";
				cout << "----------------------------------" << endl;
				for( int i = 0; i < 6 ; ++i ) cout << currentSix[i];
				cout << endl;
			}
			break;
		case 3:
			for( int i = 0; i < 6 ; ++i ) cout << currentSix[i];
			cout << endl;
			for( int i = 0; i < 6 ; ++i ) cout << consec6[i];
			cout << endl;

			for( int i = 2; i < 4 ; ++i ) {
				if( result[i] == '#' ) continue;
				if( consec6[i] < CONSEC_CORRECT ) {
					if( currentSix[i] == '#') {
						currentSix[i] = result[i];
						consec6[i] = 1;
					}
					else if( currentSix[i] == result[i] )
						consec6[i] += 1;
					else {
						currentSix[i] = result[i];
						consec6[i] = 1;
					}
				}
			}
			for( int i = 0; i < 6 ; ++i ) {
				if( consec6[i] < CONSEC_CORRECT ) Stop = false;
				result[i] = currentSix[i];
			}

			for( int i = 0; i < 6 ; ++i ) cout << currentSix[i];
			cout << endl;
			for( int i = 0; i < 6 ; ++i ) cout << consec6[i];
			cout << endl;
			cout << "----------------------------------" << endl;

			if(Stop) {
				cout << "predicted category: " << category << endl;
				cout << "predicted result: \n";
				cout << "----------------------------------" << endl;
				for( int i = 0; i < 6 ; ++i ) cout << currentSix[i];
				cout << endl;
			}
			break;
		default:
			cout << "error in stop()!\n";
			exit(1);
	}
	if(Stop) {
		currentSeven = {'#', '#', '#', '#', '#', '#', '#'};
		currentSix = {'#', '#', '#', '#', '#', '#'};
		consec7 = {0, 0, 0, 0, 0, 0, 0};
		consec6 = {0, 0, 0, 0, 0, 0};
	}
	return Stop;
}

template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
	os << "[";
	for (int i = 0; i < v.size(); ++i)
	{
	    os << v[i];
	    if (i != v.size() - 1)
	    {
	        os << ", ";
	    }
	}
	os << "]";
	return os;
}

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
ostream& operator<<(ostream& os, const ONNXTensorElementDataType& type)
{
	switch (type)
	{
	    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
	        os << "undefined";
	        break;
	    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
	        os << "float";
	        break;
	    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
	        os << "uint8_t";
	        break;
	    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
	        os << "int8_t";
	        break;
	    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
	        os << "uint16_t";
	        break;
	    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
	        os << "int16_t";
	        break;
	    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
	        os << "int32_t";
	        break;
	    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
	        os << "int64_t";
	        break;
	    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
	        os << "std::string";
	        break;
	    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
	        os << "bool";
	        break;
	    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
	        os << "float16";
	        break;
	    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
	        os << "double";
	        break;
	    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
	        os << "uint32_t";
	        break;
	    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
	        os << "uint64_t";
	        break;
	    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
	        os << "float real + float imaginary";
	        break;
	    case ONNXTensorElementDataType::
	        ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
	        os << "double real + float imaginary";
	        break;
	    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
	        os << "bfloat16";
	        break;
	    default:
	        break;
	}
	return os;
}
