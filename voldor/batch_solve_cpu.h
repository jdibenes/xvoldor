
#pragma once

#include <atomic>

struct job_descriptor
{
    int const* rng;
    int id;
    int start;
    int end;
    int point_count;
    int sample_size;
};

typedef void(*batch_callback)(void*, job_descriptor&, std::atomic<int>&);

int batch_solve(int jobs, int workers, batch_callback f, void* data, int point_count, int sample_size);
