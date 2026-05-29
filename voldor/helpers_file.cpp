
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <filesystem>
#include <stdio.h>
#include <string>
#include <stdexcept>
#include <Eigen/Eigen>
#include "helpers_eigen.h"
#include "helpers_lock.h"

int load_file(char const* filename, void* buffer, int offset, int count)
{
    FILE* f = fopen(filename, "rb");
    if (!f) { throw std::runtime_error(std::string("Failed to open file: ") + filename); }
    Cleaner c([=]() { fclose(f); });
    int bytes;
    if (count < 0)
    {
    if (fseek(f, 0, SEEK_END) != 0) { throw std::runtime_error(std::string("Error reading file: ") + filename); }
    int total = static_cast<int>(ftell(f));
    if (total < 0) { throw std::runtime_error(std::string("Error reading file: ") + filename); }
    bytes = total - offset;
    if (bytes < 0) { throw std::runtime_error(std::string("Offset beyond file size: ") + filename); }
    }
    else
    {
    bytes = count;
    }
    if (!buffer) { return bytes; }
    if (fseek(f, offset, SEEK_SET) != 0) { throw std::runtime_error(std::string("Error reading file: ") + filename); }
    if (fread(buffer, 1, bytes, f) != bytes) { throw std::runtime_error(std::string("Error reading data: ") + filename); }
    return bytes;
}

void load_flow(char const* filename, float* buffer, int buffer_index, int* width, int* height, bool strict, int check_width, int check_height)
{
    FILE* f = fopen(filename, "rb");
    if (!f) { throw std::runtime_error(std::string("Failed to open file: ") + filename); }
    Cleaner c([=]() { fclose(f); });
    float m;
    if (fread(&m, sizeof(float), 1, f) != 1) { throw std::runtime_error(std::string("Error reading magic: ") + filename); }
    if (strict && (m != 202021.25f)) { throw std::runtime_error(std::string("Bad magic: ") + filename); }
    int w;
    int h;
    if (fread(&w, sizeof(int), 1, f) != 1) { throw std::runtime_error(std::string("Error reading width: ") + filename); }
    if (fread(&h, sizeof(int), 1, f) != 1) { throw std::runtime_error(std::string("Error reading height: ") + filename); }
    if ((w < 0) || (h < 0)) { throw std::runtime_error(std::string("Unsupported dimensions: ") + filename); }
    int count = w * h * 2;
    if (count < 0) { throw std::runtime_error(std::string("Unsupported size: ") + filename); }
    if ((check_width  >= 0) && (check_width  != w)) { throw std::runtime_error(std::string("Bad width: ") + filename); }
    if ((check_height >= 0) && (check_height != h)) { throw std::runtime_error(std::string("Bad height: ") + filename); }
    if (width)  { *width  = w; }
    if (height) { *height = h; }
    if (!buffer) { return; }
    float* dst = buffer + (buffer_index * count);
    if (fread(dst, sizeof(float), count, f) != count) { throw std::runtime_error(std::string("Error reading data: ") + filename); }
    if (!strict) { return; }
    for (int i = 0; i < count; ++i) { if (!std::isfinite(dst[i])) { throw std::runtime_error(std::string("Data has non-finite elements: ") + filename); } }
}

void load_pose_hl2ss(char const* filename, float* buffer)
{
    FILE* f = fopen(filename, "rb");
    if (!f) { throw std::runtime_error(std::string("Failed to open file: ") + filename); }
    Cleaner c([=]() { fclose(f); });
    int const total = 16;
    Eigen::Matrix<float, 4, 4> pose;
    if (fread(pose.data(), sizeof(float), total, f) != total) { throw std::runtime_error(std::string("Error reading data: ") + filename); }
    Eigen::Matrix<float, 4, 4> hl2_to_opencv
    {
        {1,  0,  0,  0},
        {0, -1,  0,  0},
        {0,  0, -1,  0},
        {0,  0,  0,  1}
    };
    matrix_to_buffer(Eigen::Matrix<float, 4, 4>{hl2_to_opencv* pose* hl2_to_opencv}, buffer);
}

void load_window_flow(char const* path, char const* pattern, int start, int stop, float* buffer, bool strict, int width, int height)
{
    for (int i = start; i < stop; ++i)
    {
    int n = snprintf(nullptr, 0, pattern, i);
    std::unique_ptr<char[]> name = std::make_unique<char[]>(n + 1);
    sprintf(name.get(), pattern, i);
    std::string filename = (std::filesystem::path(path) / std::filesystem::path(name.get())).string();
    load_flow(filename.c_str(), buffer, i - start, &width, &height, strict, width, height);
    }
}

void load_window_pose_hl2ss(char const* path, char const* pattern, int start, int stop, float* buffer)
{
    Eigen::Matrix<float, 4, 4> pose_a;
    Eigen::Matrix<float, 4, 4> pose_b;
    for (int i = start; i < stop; ++i)
    {
    int n = snprintf(nullptr, 0, pattern, i);
    std::unique_ptr<char[]> name = std::make_unique<char[]>(n + 1);
    sprintf(name.get(), pattern, i);
    std::string filename = (std::filesystem::path(path) / std::filesystem::path(name.get())).string();
    pose_b = pose_a;
    load_pose_hl2ss(filename.c_str(), pose_a.data());
    if (i > start) { matrix_to_buffer(Eigen::Matrix<float, 4, 4>{pose_a.inverse()* pose_b}, buffer + ((i - (start + 1)) * 16)); }    
    }
}
