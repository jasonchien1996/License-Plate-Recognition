/*
nvcc alpr.cpp util.cpp car.cpp imgproc.cu -o alpr.out -std c++14 -gencode=arch=compute_72,code=sm_72 -I/usr/local/include/opencv4 -I/usr/local/include/onnxruntime -I/home/pomchi/Desktop/alpr/darknet/include -L/usr/local/lib -lopencv_core -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_dnn -lopencv_cudaimgproc -lopencv_highgui -lopencv_cudaarithm -lopencv_cudawarping -lopencv_cudafilters -ldarknet -lonnxruntime
*/

#include "car.hpp"
#include "util.hpp"
#include "imgproc.cuh"
#include "darknet.h"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

//for onnxruntime
#define useCUDA true
#define instanceName "plate-detector-inference"
#define modelFilepath "./checkpoints/custom-416.onnx"


//for darknet
#define OCR_TRESHOLD .3f

#define iou_threshold .45f
#define conf_threshold .5f
#define CONSEC_CORRECT 5

using namespace cv;
using namespace std;
using namespace util;
using namespace std::chrono;

char OCR_WEIGHTS[] = "ocr/ocr-net.weights";
char OCR_NETCFG[]  = "ocr/ocr-net.cfg";
char OCR_DATASET[] = "ocr/ocr-net.data";

template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v);
ostream& operator<<(ostream& os, const ONNXTensorElementDataType& type);
bool earlyStop( Car&, Car& );

int main() {
	VideoCapture capture( "./video/video (3).mp4" );
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
	// ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
	// removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
	// (Includes level 1 + more complex optimizations like node fusions)
	// ORT_ENABLE_ALL -> To Enable All possible optimizations
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
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

	Car plate;
	Car currentPlate;
	vector<Mat> images;
	while(1) {
		auto start = high_resolution_clock::now();

		Mat imageBGR, resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
		//capture >> imageBGR;
		imageBGR = cv::imread("./picture/4.jpeg");
		
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
		cout << "output tensor shape " << info.GetShape() << endl;

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
		
		if(boxes.size() == 0) continue;
		
		nms(boxes, iou_threshold);
		
		if(boxes.size() > 1) {
			cout << "#(box): " << boxes.size() << endl;			
			continue;
		}
		
		Mat croppedBGR, croppedRGB;
		util::box &bb = boxes[0];
		//cout << "[" << bb.tl[0] << "," << bb.tl[1] << "] [" << bb.br[0] << "," << bb.br[1] << "]" << endl;
		bb.tl[0] = (bb.tl[0] < 0.f) ? 0.f:bb.tl[0];
		bb.tl[1] = (bb.tl[1] < 0.f) ? 0.f:bb.tl[1];
		bb.br[0] = (bb.br[0] > 0.f) ? 1.f:bb.br[0];
		bb.br[1] = (bb.br[1] > 0.f) ? 1.f:bb.br[1];
		int x = int(bb.tl[0]*imageBGR.cols);
		int y = int(bb.tl[1]*imageBGR.rows);
		if(x-5 > 0) x -=5;
		if(y-5 > 0) y -=5;
		int width = int((bb.br[0]-bb.tl[0])*imageBGR.cols);
		int height = int((bb.br[1]-bb.tl[1])*imageBGR.rows);
		if(x+width+10 < imageBGR.cols) width+=5;
		if(y+height+10 < imageBGR.rows) height+=5;
		Rect myROI(x, y, width, height);	
		croppedBGR = imageBGR(myROI);
		//imshow("croppedBGR", croppedBGR);
		//waitKey(0);

		cvtColor(croppedBGR, croppedRGB, ColorConversionCodes::COLOR_BGR2RGB);

		Mat a,b;//create dummy Mat for space
		images.push_back(a);
		images.push_back(b);
		if( process(croppedRGB, &images) ) {
			for( cv::Mat &img : images) {		
				cv::imshow("img", img);
				cv::waitKey(0);
				
				image im = mat_to_image(img);
				string OCRresult = ocr( ocr_net, ocr_meta, im, OCR_TRESHOLD );

				if( OCRresult.length() < 6 || OCRresult.length() > 8 ) continue;
				//cout << "Darknet: " << OCRresult << endl;
				
				int category = -1;

				bool imply_category1 = plate.recognize( OCRresult, category );
				cout << "category: " << category << endl; 
				cout << "isCat1: " << imply_category1 << endl;
				cout << "OCRresult: " << OCRresult << endl;
				cout << "----------------------------------"<< endl;

				if ( OCRresult == "") continue;
				int conf = 0;
				plate.setCounter( OCRresult, category, conf, imply_category1 );
				currentPlate.setCounter( OCRresult, category, conf, imply_category1 );			
			}
			
			if( earlyStop(plate, currentPlate) ) {	
				//cout << "early predicted category: " << plate.getDominantCategory() << endl;
				//cout << "early predicted result:" << plate.getPlate() << endl;
				//break;
			}
		}
		images.clear();

		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		printf("%4.3fseconds\n", duration.count()*0.000001);
		cout << "=================================="<< endl;	
	} 	
	capture.release();    
	return 0;
}

bool earlyStop(Car &plate, Car &current_plate) {
	bool stop = true;
	int category = plate.getDominantCategory();
	int current_category = current_plate.getDominantCategory();
	if( category != -1 && category == current_category ) {
		string previous = plate.getPlate();
		string current = current_plate.vote();
		//cout << previous << " old" << endl;
		//cout << current << " new" << endl;

		array<unsigned, 7> consec7 = plate.getConsec7();
		array<unsigned, 6> consec6 = plate.getConsec6();
		if( category == 0 ) {			
			for( int i = 0; i < 7 ; ++i ) {
				//cout << consec7[i];
				if( current[i] == '#' ) continue;
				if( consec7[i] < CONSEC_CORRECT ) {
					if( previous[i] == '#') {
						plate.setPlate(i, current[i]);
						plate.resetConsecutive(i);
						consec7[i] = 1;
					}
					else if( previous[i] == current[i] ) {
						plate.increaseConsecutive(i);
						consec7[i] += 1;
					}
					else {
						if( current[i] != '#' ) {
							plate.setPlate(i, current[i]);
							plate.resetConsecutive(i);
							consec7[i] = 1;
						}
					}
					if( consec7[i] < CONSEC_CORRECT ) stop = false;
				}
			}
		}
		else if( category == 1 || category == 2) {			
			for( int i = 0; i < 6 ; ++i ) {
				//cout << consec6[i];
				if( current[i] == '#' ) continue;
				if( consec6[i] < CONSEC_CORRECT ) {
					if( previous[i] == '#') {
						plate.setPlate(i, current[i]);
						plate.resetConsecutive(i);
						consec6[i] = 1;
					}
					else if( previous[i] == current[i] ) {
						plate.increaseConsecutive(i);
						consec6[i] += 1;
					}
					else {
						if( current[i] != '#' ) {
							plate.setPlate(i, current[i]);
							plate.resetConsecutive(i);
							consec6[i] = 1;
						}
					}
					if( consec6[i] < CONSEC_CORRECT ) stop = false;
				}
			}
		}
		else {
			cout << "\nerror in earlyStop()\n";
			exit(1);
		}
		//cout << endl;
		consec7 = plate.getConsec7();
		consec6 = plate.getConsec6();
		if( category == 0 ) {
			for( int i = 0; i < 7; ++i ) {    	
				//cout << consec7[i];
			}
		}
		else if( category == 1 || category == 2) {
			for( int i = 0; i < 6; ++i ) {    	
				//cout << consec6[i];
			}
		}
		//cout << endl;
	}
	return stop;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
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
std::ostream& operator<<(std::ostream& os, const ONNXTensorElementDataType& type)
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
