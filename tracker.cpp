// tracker.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "utils.h"
#include "select_faces.h"
#include "tracker.h"

using namespace cv;
using namespace std;

tracker_t::tracker_t(Mat frame, Rect face, CascadeClassifier haar_face_cascade, string win_name, double rate) :
_face(face),
_haar_face_cascade(haar_face_cascade),
_win_name(win_name)
{
	remove(_win_name.c_str());//delete a old result for the new one
	
	if (!_video.open(_win_name, CV_FOURCC('X', 'V', 'I', 'D'), rate, _face.size(), true))
		cerr << "error open video writer" << endl;

	cvtColor(frame, _gray, CV_BGR2GRAY);
	find_good_features_to_track();//find features points
}

void tracker_t::image_process(Mat frame)
{
	cvtColor(frame, _gray, CV_BGR2GRAY);

	if (!calc_optical_flow() || _points_prev.size() < 60)
	{
		find_face_to_track(frame);//try to find the same face that was lost
		return;
	}

	estimate_rigid_transform(frame);

	_video.write(_result);
	swap(_points_prev, _points);
	swap(_gray_prev, _gray);
}

void tracker_t::show_result(Mat frame)
{
	draw_poly_lines(frame, _face_rect_points, Scalar(255, 255, 255));
	//draw_points(frame, _points_prev, Scalar(255, 255, 0));
	imshow(_win_name, _result);
}

void tracker_t::find_face_to_track(Mat frame)
{
	auto faces = detect_faces(frame, _haar_face_cascade);
	_face = find_best_match_rect(_face_boundery, faces);//try to find the same face that was lost

	find_good_features_to_track();//find features points
}

bool tracker_t::calc_optical_flow()
{
	vector<uchar> status; // status of tracked features
	vector<float> err; // error in tracking

	if (_points_prev.size() == 0)
		return false;

	calcOpticalFlowPyrLK(_gray_prev, _gray, _points_prev, _points, status, err);

	if (countNonZero(status) < status.size() * 0.9)
	{
		cout << "refresh" << endl;
		return false;
	}

	filter_points([&](int i) -> bool
	{
		return status[i] != 0;
	});

	return true;
}

void tracker_t::filter_points(function<bool(int i)> pred)
{
	int k = 0;
	for (size_t i = 0; i < _points.size(); i++)
		if (pred(i))
		{
			_points[k] = _points[i];
			_points_prev[k++] = _points_prev[i];
		}

	_points.resize(k);
	_points_prev.resize(k);
}

void tracker_t::estimate_rigid_transform(Mat frame)
{
	Mat nrt2x3 = estimateRigidTransform(_points_prev, _points, true); // false = rigid transform, no scaling/shearing

	// in rare cases no transform is found. We'll just use the last known good transform.
	if (nrt2x3.data == NULL)
		return;

	Mat_<float> nrt3x3 = Mat_<float>::eye(3, 3);
	nrt2x3.copyTo(nrt3x3.rowRange(0, 2));

	perspectiveTransform(_face_rect_points, _face_rect_points, nrt3x3);//move the rect to it's new postion in the new frame

	_face_boundery = boundingRect(_face_rect_points);
	fix_rect_bounds(_face_boundery, frame.size());

	filter_points([&](int i) -> bool
	{
		return _face_boundery.contains(_points[i]);
	});

	//for video stabilization we apply a transform to the first frame
	_rigid_transform *= nrt3x3;
	Mat invTrans = _rigid_transform.inv(0);
	warpPerspective(frame, _result, invTrans, frame.size());
	auto rect_points = _face_rect_points;
	perspectiveTransform(rect_points, rect_points, invTrans);

	//crop result
	Rect r = boundingRect(rect_points);
	r.width = _face.width;
	r.height = _face.height;
	fix_rect_bounds(r, frame.size());
	_result = _result(r);
}

void tracker_t::find_good_features_to_track()
{
	//create a nask for the face
	Mat mask = Mat::zeros(_gray.size(), CV_8U);
	auto roi = Mat(mask, _face);
	roi = Scalar(255, 255, 255);

	_face_boundery = boundingRect(mask);
	goodFeaturesToTrack(_gray, _points_prev, 300, 0.001, 5, mask);

	get_rect_points(_face, _face_rect_points);

	_rigid_transform = Mat_<float>::eye(3, 3);
	_gray.copyTo(_gray_prev);
	_points.clear();
}

