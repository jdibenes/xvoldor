
#pragma once

#include <vector>

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

struct job_result
{
    int const id;
    int const start;
    int const end;
    int const valid;
};

typedef void(*batch_callback)(job_descriptor&);

std::vector<job_result> batch_solve(int jobs, int workers, batch_callback f, void* inputs, int point_count, int sample_size, bool unique, void* output);
int batch_finalize(std::vector<job_result> const& jr, void* output, int solution_elements);

int const* get_sample_indices(job_descriptor const& jd, int index);

bool is_valid_solution_6(float const* r, float const* t);
bool is_valid_solution_f(float const* f);

void put_solution_6(job_descriptor& jd, float* base, float const* r, float const* t);
void put_solution_f(job_descriptor& jd, float* base, float const* f);
