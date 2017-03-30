#include "stdafx.h"
#include "select_faces.h"
#include "tracker.h"
#include "utils.h"

using cv::VideoWriter;
using cv::VideoCapture;
using cv::waitKey;
using std::unique_ptr;
using std::stringstream;
using std::make_unique;
using std::move;
using std::cerr;
using std::endl;

int _tmain(int argc, _TCHAR* argv[])
{
	//init the haar_cascade
	char fileName[MAX_PATH];
	ExpandEnvironmentStringsA("%OPENCV_DIR%/../../etc/haarcascades/haarcascade_frontalface_alt.xml", fileName, MAX_PATH);
	CascadeClassifier haar_face_cascade;
	haar_face_cascade.load(fileName);

	VideoCapture cap;
	if (argc == 2)
		cap.open(argv[1]);
	else
		cap.open(0);

	if (!cap.isOpened())
	{
		cerr << "unable to open VideoCapture" << endl;
		return 0;
	}
		

	Mat frame;
	auto selected_faces = select_facesV1(cap, haar_face_cascade, frame);//user can select which face to track

	if (selected_faces.size() == 0)
		return 0;

	//create tracker for each face that was selected
	double rate = cap.get(CV_CAP_PROP_FPS);
	int i = 0;
	vector<unique_ptr<tracker_t>> trackers;
	for (auto face : selected_faces)
	{
		stringstream ss;
		ss << "result" << i++ <<".avi";
		auto tracker = make_unique<tracker_t>(frame, face, haar_face_cascade, ss.str(), rate);
		trackers.push_back(move(tracker));
	}
	
	//run until file end or exit esc key
	while (true)
	{
		cap >> frame;
		if (frame.empty())
			exit(0);

#pragma omp parallel for
		for (int i = 0; i < (int)trackers.size(); ++i)
			trackers[i]->image_process(frame);

		for (auto &tracker : trackers)
			tracker->show_result(frame);

		imshow("output", frame);
		//exit esc key, all the other keys pause the loop
		auto key = waitKey(33);
		if (key == 27)
			break;
		else if (key != -1)
			waitKey(0);
	}
	return 0;
}