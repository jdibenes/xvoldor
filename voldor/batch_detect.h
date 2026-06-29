
#pragma once

#include <opencv2/calib3d.hpp>

int batch_cpu_detect_planar(cv::Point3f const* p2z_1, cv::Point3f const* p2z_2, int point_count, cv::Mat const& K, int solver, int poses_to_sample, float* poses, int workers, bool unique);
