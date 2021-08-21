#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include "car.h"
#include "imgproc.cu"

using namespace std;
using namespace std::chrono;
/*
nvcc main.cpp libimgproc.so -o main.out -std c++11 -gencode=arch=compute_72,code=sm_72 -I/usr/local/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_cudaimgproc -lopencv_highgui -lopencv_cudaarithm -lopencv_cudawarping -lopencv_cudafilters -llept -ltesseract 
*/

const int CONSEC_CORRECT = 5;

bool earlyStop(Car&, Car&);

int main() {
	/*
    cv::VideoCapture capture("video (3).mp4");
    if (!capture.isOpened()){
        printf("Unable to open file!");
        return 0;
    }
	
	tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
	api->SetVariable("tessedit_char_whitelist","0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    // Initialize tesseract-ocr with English, without specifying tessdata path
    if (api->Init(NULL, "eng")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }
	*/
	Car plate;
	Car currentPlate;
    while(1) {
		auto start = high_resolution_clock::now();
		
        //capture >> cpu;
		cv::Mat source = cv::imread("./picture/1.jpg");
        cv::Mat a,b,c;//for space
		vector<cv::Mat> images = {a, b, c};
		if ( process(source, &images) ) {
			for ( cv::Mat img : images) {
				/*
				api->SetImage((uchar *)img.data, img.cols, img.rows, img.channels(), img.step1());
				Boxa* boxes = api->GetComponentImages(tesseract::RIL_TEXTLINE, true, NULL, NULL);
				for ( int i = 0; i < boxes->n; ++i ) {
					BOX* box = boxaGetBox(boxes, i, L_CLONE);
					api->SetRectangle(box->x, box->y-1, box->w, box->h+1);
					char *text = api->GetUTF8Text();
					int conf = api->MeanTextConf();
					if(conf < 20) continue;
					printf("OCR output: %s", text);
					delete [] text;
				}
				*/
				int category = -1;
				int conf = 100;
				string OCRresult = "ABC1234";

				bool imply_category1 = plate.recognize(OCRresult, category);
				if ( OCRresult == "") continue;

				plate.setCounter(OCRresult, category, conf, imply_category1);
				currentPlate.setCounter(OCRresult, category, conf, imply_category1);
				/*
				cout << endl << "category: " << category << endl; 
				cout << "isCat1: " << imply_category1 << endl;
				cout << "OCRresult: " << OCRresult << endl;
				*/
			}

			if ( earlyStop(plate, currentPlate) ) {	
		        cout << "early predicted category: " << plate.getDominantCategory() << endl;
		        cout << "early predicted result:" << plate.getPlate() << endl;
				//break;
			}		
		}
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        printf("%4.3fseconds\n", duration.count()*0.000001);
    }
 	
    //capture.release();
	//api->End();
    //delete api;
    
    return 0;
}

bool earlyStop(Car &plate, Car &current_plate){
	bool stop = true;
    int category = plate.getDominantCategory();
	int current_category = current_plate.getDominantCategory();
    if ( category != -1 && category == current_category ) {
        string previous = plate.getPlate();
        string current = current_plate.vote();
        cout << previous << " old" << endl;
		cout << current << " new" << endl;

		array<unsigned, 7> consec7 = plate.getConsec7();
		array<unsigned, 6> consec6 = plate.getConsec6();
		if ( category == 0 ) {			
		    for ( int i = 0; i < 7 ; ++i ) {
				cout << consec7[i];
				if ( current[i] == '#' ) continue;
		        if ( consec7[i] < CONSEC_CORRECT ) {
					if ( previous[i] == '#') {
						plate.setPlate(i, current[i]);
						plate.resetConsecutive(i);
						consec7[i] = 1;
					}
		            else if ( previous[i] == current[i] ) {
		                plate.increaseConsecutive(i);
						consec7[i] += 1;
					}
		            else {
		                if ( current[i] != '#' ) {
		                    plate.setPlate(i, current[i]);
		                    plate.resetConsecutive(i);
							consec7[i] = 1;
						}
					}
		            if ( consec7[i] < CONSEC_CORRECT )
		                stop = false;
				}
			}
		}
		else if ( category == 1 || category == 2) {			
		    for ( int i = 0; i < 6 ; ++i ) {
				cout << consec6[i];
				if ( current[i] == '#' ) continue;
		        if ( consec6[i] < CONSEC_CORRECT ) {
					if ( previous[i] == '#') {
						plate.setPlate(i, current[i]);
						plate.resetConsecutive(i);
						consec6[i] = 1;
					}
		            else if ( previous[i] == current[i] ) {
		                plate.increaseConsecutive(i);
						consec6[i] += 1;
					}
		            else {
		                if ( current[i] != '#' ) {
		                    plate.setPlate(i, current[i]);
		                    plate.resetConsecutive(i);
							consec6[i] = 1;
						}
					}
		            if ( consec6[i] < CONSEC_CORRECT )
		                stop = false;
				}
			}
		}
		else {
			cout << "\nerror in earlyStop()\n";
			exit(1);
		}
		cout << endl;
		consec7 = plate.getConsec7();
		consec6 = plate.getConsec6();
		if ( category == 0 ) {
			for ( int i = 0; i < 7; ++i )        	
				cout << consec7[i];
		}
		else if ( category == 1 || category == 2) {
			for ( int i = 0; i < 6; ++i )        	
				cout << consec6[i];
		}
		cout << endl;
	}
    return stop;
}
