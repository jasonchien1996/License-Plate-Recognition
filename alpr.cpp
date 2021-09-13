/*
nvcc alpr.cpp ocr.cpp imgproc.cu -o alpr.out -std c++11 -gencode=arch=compute_72,code=sm_72 -I/usr/local/include/opencv4 -I/home/pomchi/Desktop/alpr/darknet/include -lopencv_core -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_cudaimgproc -lopencv_highgui -lopencv_cudaarithm -lopencv_cudawarping -lopencv_cudafilters -ldarknet
*/

#include "ocr.hpp"
#include "car.hpp"
#include "darknet.h"
#include "imgproc.cuh"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#define OCR_TRESHOLD .3f
#define CONSEC_CORRECT 5

using namespace std;
using namespace std::chrono;

char OCR_WEIGHTS[] = "ocr/ocr-net.weights";
char OCR_NETCFG[]  = "ocr/ocr-net.cfg";
char OCR_DATASET[] = "ocr/ocr-net.data";

bool earlyStop( Car&, Car& );

int main() {
	
	cv::VideoCapture capture( "./video/video (3).mp4" );
	if( !capture.isOpened() ) {
		printf( "Unable to open file!" );
		return 0;
	}		
		
	network *ocr_net = load_network( OCR_NETCFG, OCR_WEIGHTS, 0 );
	metadata ocr_meta = get_metadata( OCR_DATASET );
	set_batch_network(ocr_net, 1);

	Car plate;
	Car currentPlate;
	vector<cv::Mat> images;		
	
	while(1) {
		auto start = high_resolution_clock::now();
		/*
		cv::Mat source;
		capture >> source;
		*/
	
		cv::Mat source = cv::imread("./picture/4.jpeg");

		/*do inference*/		
		image im = mat_to_image(source);		
		string OCRresult = ocr( ocr_net, ocr_meta, im, OCR_TRESHOLD );
		
		cv::Mat a,b,c;//create dummy Mat for space
		images.push_back(a);
		images.push_back(b);
		images.push_back(c);
		if( process(source, &images) ) {
			for( cv::Mat &img : images) {
				if( img.type() == 0 ) continue;
				/*
				cv::imshow("img", img);
				cv::waitKey(0);
				*/
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
