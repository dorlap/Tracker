#include "stdafx.h"
#include "utils.h"

vector<Rect> detect_faces(Mat frame, CascadeClassifier haar_face_cascade)
{
	Mat gray;
	cvtColor(frame, gray, CV_BGR2GRAY);

	vector<Rect> faces;
	haar_face_cascade.detectMultiScale(gray, faces, 1.2, 6, 0, Size(45, 45));

	return faces;
}

Rect find_best_match_rect(Rect rect, vector<Rect> const &rects)
{
	//the center point of the last face we had
	Point center_point;
	center_point.x = (rect.tl().x + rect.br().x) / 2;
	center_point.y = (rect.tl().y + rect.br().y) / 2;

	//find the closest center point of the new detected faces
	float min_diff = FLT_MAX;
	Rect ret;
	for (auto r : rects)
	{
		Point2f p((float)(r.tl().x + r.br().x) / 2, (float)(r.tl().y + r.br().y) / 2);
		float diff = sqrt(pow(center_point.x - p.x, 2) + pow(center_point.y - p.y, 2));

		if (min_diff > diff)
		{
			min_diff = diff;
			ret = r;
		}
	}

	return ret;
}

void draw_poly_lines(Mat frame, vector<Point2f> const &points, Scalar color)
{
	for (size_t i = 0; i < points.size(); ++i)
		line(frame, points[i], points[(i + 1) % points.size()], color);
}

void draw_points(Mat frame, vector<Point2f> const &points, Scalar color)
{
	for (auto point : points)
		circle(frame, point, 3, color, -1);
}

void get_rect_points(Rect const &rect, vector<Point2f> &points)
{
	vector<Point2f> vec{
		Point(rect.tl().x, rect.tl().y),
		Point(rect.br().x, rect.tl().y),
		Point(rect.br().x, rect.br().y),
		Point(rect.tl().x, rect.br().y)
	};

	swap(points, vec);
}

void fix_rect_bounds(Rect &rect, Size size)
{
	if (rect.x < 0)
		rect.x = 0;
	else if (rect.x + rect.width > size.width)
		rect.width = size.width - rect.x;

	if (rect.y < 0)
		rect.y = 0;
	else if (rect.y + rect.height > size.height)
		rect.y = size.height - rect.y;
}