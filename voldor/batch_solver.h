
#pragma once

#include <opencv2/calib3d.hpp>

int batch_cpu_solver_p4p(cv::Point3f const* p3d_1, cv::Point2f const* p2k_2,                           int point_count, cv::Mat const& K, int solver, int poses_to_sample, float* poses,                    int workers, bool unique);
int batch_gpu_solver_p4p(cv::Point3f const* p3d_1, cv::Point2f const* p2k_2,                           int point_count, cv::Mat const& K, int solver, int poses_to_sample, float* poses);
int batch_cpu_solver_gpm(cv::Point3f const* p2z_1, cv::Point3f const* p2z_2,                           int point_count, cv::Mat const& K, int solver, int poses_to_sample, float* poses,                    int workers, bool unique);
int batch_cpu_solver_tft(cv::Point3f const* p2z_1, cv::Point3f const* p2z_2, cv::Point3f const* p2z_3, int point_count, cv::Mat const& K, int solver, int poses_to_sample, float* poses, float* next_pool,  int workers, bool unique, float threshold);
int batch_cpu_solver_rnp(cv::Point3f const* p3d_1, cv::Point2f const* p2k_2,                           int point_count, cv::Mat const& K, int solver, int poses_to_sample, float* poses, float* velocities, int workers, bool unique, bool direction, float r0, int max_iterations);
int batch_cpu_solver_ppf(cv::Point3f const* p3d_1, cv::Point2f const* p2k_2,                           int point_count, cv::Mat const& K, int solver, int poses_to_sample, float* poses, float* focals,     int workers, bool unique, bool same);
