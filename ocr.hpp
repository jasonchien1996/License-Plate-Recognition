#include "darknet.h"
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct ele{
	char *name;
	float prob;
	float x;
	float y;
	float w;
	float h;
};


struct label{
	char cl;
	float tl[2];
	float br[2];	
	float prob;
};

image mat_to_image(const Mat&);
string ocr(network *ocr_net, metadata &ocr_meta, image &im, float ocr_threshold);
