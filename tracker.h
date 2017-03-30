#pragma once
#include "utils.h"
#include <string>
#include <functional>

using cv::VideoWriter;
using std::string;
using std::function;

class tracker_t
{
public:
	tracker_t(Mat frame, Rect face, CascadeClassifier haar_face_cascade, string win_name, double rate);

	void image_process(Mat frame);
	void show_result(Mat frame);
	
private:
	void find_face_to_track(Mat frame);
	bool calc_optical_flow();
	void filter_points(function<bool(int i)> pred);
	void estimate_rigid_transform(Mat frame);
	void find_good_features_to_track();

private:
	Mat _gray; // current gray-level image
	Mat _gray_prev; // previous gray-level image
	Mat _rigid_transform;
	Mat _result;
	Rect _face;
	Rect _face_boundery;
	vector<Point2f> _points;
	vector<Point2f> _points_prev;
	vector<Point2f> _face_rect_points;
	CascadeClassifier _haar_face_cascade;
	string _win_name;
	VideoWriter _video;
};