
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

int batch_solve(int jobs, int workers, batch_callback f, void* inputs, int point_count, int sample_size, bool unique, void* output, int solution_elements);

int const* get_sample_indices(job_descriptor const& jd, int index);

bool is_valid_solution_6(float const r[3], float const t[3]);
bool is_valid_solution_7(float const r[3], float const t[3], float f);

void put_solution_6(job_descriptor& jd, float const r[3], float const t[3]);
void put_solution_7(job_descriptor& jd, float const r[3], float const t[3], float f);
