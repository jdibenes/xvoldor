
#include "utils.h"

cv::Mat vis_flow(cv::Mat flow, float mag_scale) {
	cv::Mat flow_xy[2];
	cv::Mat mag, angle;
	split(flow, flow_xy);
	cv::cartToPolar(flow_xy[0], flow_xy[1], mag, angle, true);
	if (mag_scale <= 0)
		normalize(mag, mag, 0, 1, cv::NORM_MINMAX);
	else
		mag /= mag_scale;
	cv::Mat dst;
	std::vector<cv::Mat> src{ angle, mag, cv::Mat::ones(flow.size(), CV_32F) };
	merge(src, dst);
	cvtColor(dst, dst, cv::COLOR_HSV2BGR);
	return dst;
}


cv::Mat load_flow(const char* file_path) {
	FILE* fs = fopen(file_path, "rb");
	if (fs == NULL) {
		std::cout << file_path << " does not exist~!" << std::endl;
		throw;
	}

	float magic_num = 0;
	int w = 0, h = 0;
	fread(&magic_num, sizeof(float), 1, fs);
	assert(magic_num == 202021.25f);
	fread(&w, sizeof(int), 1, fs);
	fread(&h, sizeof(int), 1, fs);

	cv::Mat flow(cv::Size(w, h), CV_32FC2);
	fread(flow.data, sizeof(float), w*h * 2, fs);
	fclose(fs);
	return flow;
}

cv::Mat rot_mat_3d(float degx, float degy, float degz) {
	degx /= 180 * 3.14159;
	degy /= 180 * 3.14159;
	degz /= 180 * 3.14159;
	cv::Mat Rx = (cv::Mat_<float>(3, 3) <<
		1, 0, 0,
		0, cosf(degx), -sinf(degx),
		0, sinf(degx), cosf(degx));
	cv::Mat Ry = (cv::Mat_<float>(3, 3) <<
		cosf(degy), 0, sinf(degy),
		0, 1, 0,
		-sinf(degy), 0, cosf(degy));
	cv::Mat Rz = (cv::Mat_<float>(3, 3) <<
		cosf(degz), -sinf(degz), 0,
		sinf(degz), cosf(degz), 0,
		0, 0, 1);

	return Rx * Ry * Rz;
}

