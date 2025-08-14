#pragma once
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#define _USE_MATH_DEFINES

#if defined(WIN32) || defined(_WIN32) 
#define PATH_SEPARATOR '\\'
#else 
#define PATH_SEPARATOR '/'
#endif


#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>


#define div_ceil(x, y) ( (x) / (y) + ((x) % (y) > 0) )


struct Camera {
	cv::Mat F;
	cv::Mat E;

	cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
	cv::Mat K_inv = cv::Mat::eye(3, 3, CV_32F);
	cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
	cv::Mat t = cv::Mat::zeros(3, 1, CV_32F);
	cv::Mat pose_covar = cv::Mat::zeros(6, 6, CV_32F);
	//Mat _R2;
	//Mat pose_sample_mask;
	float pose_density = 0;
	int pose_sample_count = 0;
	float pose_rigidness_density = 0;
	int last_used_ms_iters = 0;
	int last_used_gu_iters = 0;

	cv::Vec6f pose6() {
		cv::Vec3f r = this->rvec();
		return cv::Vec6f(r.val[0], r.val[1], r.val[2], t.at<float>(0), t.at<float>(1), t.at<float>(2));
	}

	cv::Vec3f rvec() {
		cv::Vec3f ret;
		Rodrigues(this->R, ret);
		return ret;
	}

	void save(FILE* fs) {
		cv::Vec3f r = this->rvec();
		//           r1 r2 r3 t1 t2 t3
		fprintf(fs, "%f %f %f %f %f %f\n",
			r.val[0], r.val[1], r.val[2],
			t.at<float>(0), t.at<float>(1), t.at<float>(2));
	}

	void print_info() {
		std::cout << "pose pool size = " << this->pose_sample_count << std::endl;
		std::cout << "rigidness density = " << this->pose_rigidness_density << std::endl;
		std::cout << "pose density = " << this->pose_density << std::endl;
		std::cout << "pose covar mean scale = " << mean(pose_covar.diag())[0] << std::endl;;
		//cout << pose_covar.diag() << endl;
		std::cout << "last used meanshift iters = " << this->last_used_ms_iters << std::endl;
		std::cout << "last used gu iters = " << this->last_used_gu_iters << std::endl;
		std::cout << "pose trans mag = " << cv::norm(this->t, cv::NORM_L2) << std::endl;
		std::cout << "pose rot mag = " << cv::norm(this->rvec(), cv::NORM_L2) * 180 / 3.14159 << std::endl << std::endl;
	}
};


cv::Mat vis_flow(cv::Mat flow, float mag_scale = 0);

cv::Mat load_flow(const char* file_path);
