
#pragma once

#include <iostream>
#include <opencv2/calib3d.hpp>

struct Camera
{
	cv::Mat F;
	cv::Mat E;

	cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
	cv::Mat K_inv = cv::Mat::eye(3, 3, CV_32F);
	cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
	cv::Mat t = cv::Mat::zeros(3, 1, CV_32F);
	cv::Mat pose_covar = cv::Mat::zeros(6, 6, CV_32F);

	cv::Mat dr = cv::Mat::zeros(3, 1, CV_32F);
	cv::Mat dt = cv::Mat::zeros(3, 1, CV_32F);

	float pose_density = 0;
	int pose_sample_count = 0;
	float pose_rigidness_density = 0;
	int last_used_ms_iters = 0; // statistics
	int last_used_gu_iters = 0; // statistics

	cv::Vec6f pose6() 
	{
		cv::Vec3f r = this->rvec();
		return cv::Vec6f(r.val[0], r.val[1], r.val[2], t.at<float>(0), t.at<float>(1), t.at<float>(2));
	}

	cv::Vec3f rvec()
	{
		cv::Vec3f ret;
		Rodrigues(this->R, ret);
		return ret;
	}

	void save(FILE* fs)
	{
		cv::Vec3f r = this->rvec();
		//           r1 r2 r3 t1 t2 t3
		fprintf(fs, "%f %f %f %f %f %f\n",
			r.val[0], r.val[1], r.val[2],
			t.at<float>(0), t.at<float>(1), t.at<float>(2));
	}

	void print_info()
	{
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

void estimate_depth_closed_form(cv::Mat const& flow, cv::Mat const& K, cv::Mat const& K_inv, cv::Mat const& R, cv::Mat const& t, cv::Mat& depth, float min_depth = 1e-2f, float max_depth = 1e10f);
void estimate_camera_pose_epipolar(cv::Mat const& flow, cv::Mat const& K, cv::Mat& E, cv::Mat& R, cv::Mat& t, cv::Mat const& mask = cv::Mat(), int sampling_2d_step = 4);
void estimate_camera_focal(cv::Mat const& flow, float& fx, float& fy, float cx, float cy, int sampling_2d_step = 4);
