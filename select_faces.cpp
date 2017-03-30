#include "stdafx.h"
#include "select_faces.h"

using namespace cv;
using std::cout;

struct selected_faces_context_t
{
	Mat output;
	vector<bool> selected_faces;
	vector<Rect> all_faces;
};

void draw_selected_rect(Mat output, Rect face, bool is_selected)
{
	rectangle(output, face, is_selected ? Scalar(255, 255, 255) : Scalar(0, 0, 0), 3);
}

void select_faces_handle_mouse(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		auto ctx = (selected_faces_context_t*)userdata;
		//checks if we click in a rectangle of the face
		for (size_t i = 0; i < ctx->all_faces.size(); ++i)
		{
			auto face = ctx->all_faces[i];

			if (face.contains(Point(x, y)))
			{
				ctx->selected_faces[i] = !ctx->selected_faces[i];
				draw_selected_rect(ctx->output, face, ctx->selected_faces[i]);

				imshow("output", ctx->output);
			}
		}
	}
}

vector<Rect> select_facesV1(VideoCapture cap, CascadeClassifier haar_face_cascade, Mat &frame)
{
	vector<Rect> ret;

	while (true)
	{
		cap >> frame;
		if (frame.empty())
			break;

		imshow("output", frame);

		//esc key exit, space key pause and start face detection
		auto key = waitKey(2);
		if (key == 32)
		{
			selected_faces_context_t ctx;
			frame.copyTo(ctx.output);
			ctx.all_faces = detect_faces(frame, haar_face_cascade);//find faces
			ctx.selected_faces.resize(ctx.all_faces.size());

			//draw faces
			for (auto face : ctx.all_faces)
				draw_selected_rect(ctx.output, face, false);

			imshow("output", ctx.output);

			//user selection with a mouse
			setMouseCallback("output", select_faces_handle_mouse, &ctx);
			auto key = waitKey(0);
			setMouseCallback("output", NULL, NULL);

			//enter key accept the selected faces, esc key exit
			if (key == 27)
				break;
			else if (key == 13)
			{
				for (size_t i = 0; i < ctx.all_faces.size(); ++i)
					if (ctx.selected_faces[i])
						ret.push_back(ctx.all_faces[i]);
				break;
			}
		}
		else if (key == 27)
			break;
	}

	return ret;
}


//vector<Rect> select_faces(Mat frame, CascadeClassifier haar_face_cascade)
//{
//	selected_faces_context_t ctx;
//	frame.copyTo(ctx.output);
//	ctx.all_faces = detect_faces(frame, haar_face_cascade);
//	ctx.selected_faces.resize(ctx.all_faces.size());
//
//	for (auto face : ctx.all_faces)
//		draw_selected_rect(ctx.output, face, false);
//
//	imshow("output", ctx.output);
//	setMouseCallback("output", select_faces_handle_mouse, &ctx);
//
//	vector<Rect> ret;
//	while (true)
//	{
//		auto key = waitKey(0);
//		if (key == 27)
//			break;
//		else if (key == 13)
//		{
//			for (size_t i = 0; i < ctx.all_faces.size(); ++i)
//				if (ctx.selected_faces[i])
//					ret.push_back(ctx.all_faces[i]);
//			break;
//		}
//		else if (key == 32)
//		{
//
//		}
//	}
//
//	destroyWindow("output");
//	return ret;
//}
