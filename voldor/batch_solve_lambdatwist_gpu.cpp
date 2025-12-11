
#include <opencv2/calib3d.hpp>
#include "../gpu-kernels/gpu_kernels.h"

int batch_solve_lambdatwist_gpu(std::vector<cv::Point2f> const& pts2, std::vector<cv::Point3f> const& pts3, cv::Mat const& K, int poses_to_sample, cv::Mat& poses_pool)
{
    int n_points = (int)pts2.size();
    int poses_pool_used = 0;

    float* ret_Rs = new float[poses_to_sample * 3];
    float* ret_ts = new float[poses_to_sample * 3];

    solve_batch_p3p_lambdatwist_gpu((float*)pts3.data(), (float*)pts2.data(), ret_Rs, ret_ts, (float*)K.data, n_points, poses_to_sample);

    for (int i = 0; i < poses_to_sample; i++)
    {
        float r_sum = ret_Rs[i * 3 + 0] + ret_Rs[i * 3 + 1] + ret_Rs[i * 3 + 2];
        float t_sum = ret_ts[i * 3 + 0] + ret_ts[i * 3 + 1] + ret_ts[i * 3 + 2];

        float x_sum = r_sum + t_sum;

        if (!isfinite(x_sum)) { continue; }

        poses_pool.at<cv::Vec3f>(poses_pool_used, 0) = cv::Vec3f(ret_Rs[i * 3 + 0], ret_Rs[i * 3 + 1], ret_Rs[i * 3 + 2]);
        poses_pool.at<cv::Vec3f>(poses_pool_used, 1) = cv::Vec3f(ret_ts[i * 3 + 0], ret_ts[i * 3 + 1], ret_ts[i * 3 + 2]);
        poses_pool_used++;
    }

    delete[] ret_Rs;
    delete[] ret_ts;

    return poses_pool_used;
}
