#pragma once
#include "utils.h"

vector<Rect> select_faces(Mat frame, CascadeClassifier haar_face_cascade);
vector<Rect> select_facesV1(VideoCapture cap, CascadeClassifier haar_face_cascade, Mat &frame);