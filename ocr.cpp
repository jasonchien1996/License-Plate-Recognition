#include "ocr.hpp"
#include "darknet.h"
#include <algorithm>
#include <vector>
#include <cassert>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

bool compare_ele(const ele&, const ele&);
bool compare_label(const label&, const label&);
bool compare_tl(const label&, const label&);
void minimum(const float arr1[2], const float arr2[2], float min_arr[2]);
void maximum(const float arr1[2], const float arr2[2], float max_arr[2]);
void dknet_label_conversion(const vector<ele>&, float&, float&, vector<label>&);
void nms(vector<label>&, const float&);
float IOU_labels(const label&, const label&);
float IOU(const float tl1[2], const float br1[2], const float tl2[2], const float br2[2]);

string ocr(network *ocr_net, metadata &ocr_meta, image &im, float ocr_threshold){
	int num = 0;
	network_predict_image(ocr_net, im);
	detection *dets = get_network_boxes(ocr_net, im.w, im.h, ocr_threshold, .5f, NULL, 0, &num);

	vector<ele> res;
	for( int j = 0; j < num; ++j ) {
		for( int i = 0; i < ocr_meta.classes; ++i ) {
			if( dets[j].prob[i] > 0.f ) {
				box b = dets[j].bbox;
				ele temp = { ocr_meta.names[i], dets[j].prob[i], b.x, b.y, b.w, b.h };
				res.push_back(temp);
			}
		}
	}
	sort(res.begin(), res.end(), compare_ele);

	float w = im.w;
	float h = im.h;
	free_image(im);
	free_detections(dets, num);

	vector<label> L;
	string lp_str = "";
	if(res.size()){
		dknet_label_conversion(res, w, h, L);
		nms(L,.45f);
		sort(L.begin(), L.end(), compare_tl);
		for(const label &l : L)		
			lp_str += l.cl;
	}
	return lp_str;
}

bool compare_tl(const label &l1, const label &l2){ 
	return (l1.tl[0] < l2.tl[0]);
}

void nms(vector<label> &Labels, const float &iou_threshold){
	vector<label> SelectedLabels;
	sort(Labels.begin(), Labels.end(), compare_label);
	
	for(const label &l : Labels){
		bool non_overlap = true;
		for(const label &sel_label : SelectedLabels){
			if(IOU_labels(l,sel_label) > iou_threshold){
				non_overlap = false;
				break;
			}
		}
		if(non_overlap) SelectedLabels.push_back(l);
	}

	Labels = SelectedLabels;
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

float IOU_labels(const label &l1, const label &l2){
	return IOU(l1.tl,l1.br,l2.tl,l2.br);
}

void dknet_label_conversion(const vector<ele> &R, float &img_width, float &img_height, vector<label> &L){
	for(const ele &r : R){
		float center[2] = {r.x/img_width, r.y/img_height};
		float wh2[2] = {0.5f*r.w/img_width, 0.5f*r.h/img_height};
		label l = {*r.name, {center[0]-wh2[0], center[1]-wh2[1]}, {center[0]+wh2[0], center[1]+wh2[1]}, r.prob};
		L.push_back(l);
	}
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

bool compare_ele(const ele &e1, const ele &e2){
    return (e1.prob > e2.prob);
}

bool compare_label(const label &l1, const label &l2){
    return (l1.prob > l2.prob);
}
