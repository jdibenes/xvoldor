
#pragma once

struct job_descriptor
{
    void* inputs;
    void* output;
    int const* rng;
    int const id;
    int const start;
    int const end;
    int const point_count;
    int const sample_size;
    int valid;
};

typedef void(*batch_callback)(job_descriptor&);

int batch_solve(int jobs, int workers, batch_callback f, 
    void* inputs, int point_count, int sample_size, void* output, int solution_size_bytes);
