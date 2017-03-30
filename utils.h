#pragma once

#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include <vector>

using cv::CascadeClassifier;
using cv::Point;
using cv::Point2f;
using cv::Rect;
using cv::Scalar;
using cv::Mat;
using cv::Size;
using std::vector;
using cv::VideoCapture;

vector<Rect> detect_faces(Mat frame, CascadeClassifier haar_face_cascade);

Rect find_best_match_rect(Rect rect, vector<Rect> const &rects);

void draw_poly_lines(Mat frame, vector<Point2f> const &points, Scalar color);

void draw_points(Mat frame, vector<Point2f> const &points, Scalar color);

void get_rect_points(Rect const &rect, vector<Point2f> &points);

void fix_rect_bounds(Rect &rect, Size size);
