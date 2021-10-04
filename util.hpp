#include "darknet.h"
#include <string>
#include <opencv2/opencv.hpp>

namespace util{
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

	struct box{
		float tl[2];
		float br[2];
		float conf;
	};

	box *create_box(float ymin, float xmin, float ymax, float xmax, float conf);
	void nms(std::vector<box>&, float iou_threshold);
	image mat_to_image(const cv::Mat&);
	std::string ocr(network *ocr_net, metadata &ocr_meta, image &im, float ocr_threshold);
}
