#include "util.hpp"
#include "darknet.h"
#include <algorithm>
#include <vector>
#include <cassert>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace std::chrono;

namespace util{
	bool compare_box(const box&, const box&);
	bool compare_element(const element&, const element&);
	bool compare_tl(const box&, const box&);
	void minimum(const float arr1[2], const float arr2[2], float min_arr[2]);
	void maximum(const float arr1[2], const float arr2[2], float max_arr[2]);
	void dknet_box_conversion(const vector<element>&, float&, float&, vector<box>&);
	void nms(vector<box>&, float);
	float IOU_box(const box&, const box&);
	float IOU(const float tl1[2], const float br1[2], const float tl2[2], const float br2[2]);

	bool compare_box(const box &b1, const box &b2){
		return (b1.conf > b2.conf);
	}

	bool compare_element(const element &e1, const element &e2){
		return (e1.prob > e2.prob);
	}

	bool compare_tl(const box &l1, const box &l2){ 
		return (l1.tl[0] < l2.tl[0]);
	}

	void minimum(const float arr1[2], const float arr2[2], float min_arr[2]){
		float min1 = (arr1[0] < arr2[0]) ? arr1[0]:arr2[0];
		float min2 = (arr1[1] < arr2[1]) ? arr1[1]:arr2[1];
		min_arr[0] = min1;
		min_arr[1] = min2;
	}

	void maximum(const float arr1[2], const float arr2[2], float max_arr[2]){
		float max1 = (arr1[0] > arr2[0]) ? arr1[0]:arr2[0];
		float max2 = (arr1[1] > arr2[1]) ? arr1[1]:arr2[1];
		max_arr[0] = max1;
		max_arr[1] = max2;
	}

	void dknet_box_conversion(const vector<element> &R, float &img_width, float &img_height, vector<box> &B){
		for(const element &r : R){
			float center[2] = {r.x/img_width, r.y/img_height};
			float wh2[2] = {0.5f*r.w/img_width, 0.5f*r.h/img_height};
			box b = {*r.name, {center[0]-wh2[0], center[1]-wh2[1]}, {center[0]+wh2[0], center[1]+wh2[1]}, r.prob};
			B.push_back(b);
		}
	}

	void nms(vector<box> &Boxes, float iou_threshold){
		vector<box> SelectedBoxes;
		sort(Boxes.begin(), Boxes.end(), compare_box);
		for(const box &b : Boxes){	
			bool non_overlap = true;
			for(const box &sel_box : SelectedBoxes){
				if(IOU_box(b, sel_box) > iou_threshold){
					non_overlap = false;
					break;
				}
			}
			if(non_overlap) SelectedBoxes.push_back(b);
		}
		Boxes = SelectedBoxes;
	}

	float IOU_box(const box &b1, const box &b2){
		return IOU(b1.tl,b1.br,b2.tl,b2.br);
	}

	float IOU(const float tl1[2], const float br1[2], const float tl2[2], const float br2[2]){
		float wh1[2] = {br1[0]-tl1[0], br1[1]-tl1[1]};
		float wh2[2] = {br2[0]-tl2[0], br2[1]-tl2[1]};
		assert( wh1[0] >= .0f && wh1[1] >= .0f && wh2[0] >= .0f && wh2[1] >= .0f);

		float tmp1[2];
		float tmp2[2];
		minimum(br1, br2, tmp1);
		maximum(tl1, tl2, tmp2);
		tmp1[0] = tmp1[0]-tmp2[0];
		tmp1[1] = tmp1[1]-tmp2[1];
		tmp2[0] = 0.f;
		tmp2[1] = 0.f;
		float intersection_wh[2];
		maximum(tmp1, tmp2, intersection_wh);
		
		float intersection_area = intersection_wh[0]*intersection_wh[1];
		float area1 = wh1[0]*wh1[1];
		float area2 = wh2[0]*wh2[1];
		float union_area = area1 + area2 - intersection_area;
		return intersection_area/union_area;
	}

	void ocr(network *ocr_net, metadata &ocr_meta, image &im, float ocr_threshold, string &lp_str, int &conf){
		int num = 0;

		network_predict_image(ocr_net, im);
		detection *dets = get_network_boxes(ocr_net, im.w, im.h, ocr_threshold, .5f, NULL, 0, &num);

		vector<element> res;
		for( int j = 0; j < num; ++j ) {
			for( int i = 0; i < ocr_meta.classes; ++i ) {
				if( dets[j].prob[i] > 0.f ) {
					::box b = dets[j].bbox;
					element temp = { ocr_meta.names[i], dets[j].prob[i], b.x, b.y, b.w, b.h };
					res.push_back(temp);
				}
			}
		}

		sort(res.begin(), res.end(), compare_element);
		float w = im.w;
		float h = im.h;
		free_image(im);
		free_detections(dets, num);
		vector<box> L;
		lp_str = "";
		conf = 0;
		if(res.size()){
			dknet_box_conversion(res, w, h, L);
			nms(L,.45f);
			sort(L.begin(), L.end(), compare_tl);
			for(const box &b : L){
				lp_str += b.c;
				conf += (int)(b.conf*100);
			}
		}
	}	

	box *create_box(float ymin, float xmin, float ymax, float xmax, float conf){
		xmin = (xmin < 0.f) ? 0.f:xmin;
		ymin = (ymin < 0.f) ? 0.f:ymin;
		xmax = (xmax > 1.f) ? 1.f:xmax;
		ymax = (ymax > 1.f) ? 1.f:ymax;
		box *boxptr = (box*) new box{'\0', {xmin,ymin}, {xmax, ymax}, conf};
		return boxptr;
	}

	image mat_to_image(const Mat &m){
		int h = m.rows;
		int w = m.cols;
		int c = m.channels();
		image im = make_image(w, h, c);
		unsigned char *data = (unsigned char *)m.data;
		int step = m.step;
		int i, j, k;

		for(i = 0; i < h; ++i){
		    for(k= 0; k < c; ++k){
		        for(j = 0; j < w; ++j){
		            im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
		        }
		    }
		}
		rgbgr_image(im);
		return im;
	}
}
