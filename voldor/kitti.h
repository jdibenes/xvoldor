
#pragma once

#include <iostream>
#include <opencv2/calib3d.hpp>

struct KittiGround {
	cv::Vec3f normal = cv::Vec3f(0, 0, 0);
	float height = 0;
	float confidence = 0;
	int used_iters = 0;
	float _height_median = 0;

	void save(FILE* fs) {
		fprintf(fs, "%f %f %f %f %f\n", this->height, this->normal.val[0], this->normal.val[1], this->normal.val[2], this->confidence);
	}

	void print_info() {
		std::cout << "ground height = " << this->height << std::endl;
		std::cout << "ground normal = " << this->normal << std::endl;
		std::cout << "ground confidence = " << this->confidence << std::endl;
		std::cout << "ground used iters = " << this->used_iters << std::endl;
		std::cout << "ground height median = " << this->_height_median << std::endl;
	}
};

KittiGround
estimate_kitti_ground_plane
(
	cv::Mat depth,
	cv::Rect roi,
	cv::Mat K,
	int holo_width = 4,
	float ms_kernel_var = 0.01f
);
