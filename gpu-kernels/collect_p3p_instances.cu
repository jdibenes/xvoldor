
#include "utils.h"
#include "gpu_kernels.h"
#include "gmat.h"

#define RAND_SEED 233
#define BLOCK_WIDTH 16
#define MAX_FRAMES 16

__constant__ static float _K4[4]; // _fx, _cx, _fy, _cy
__constant__ static float _K4_inv[4]; // 1/_fx, -_cx/_fx, 1/_fy, -_cy/_fy
__constant__ static float _Rs[MAX_FRAMES][3][3];
__constant__ static float _ts[MAX_FRAMES][3];

__constant__ static int _N;
__constant__ static int _w;
__constant__ static int _h;

__constant__ static GMatf2 _d_flows_1;
__constant__ static GMatf2 _d_flows_2;
__constant__ static GMatf  _d_disparities;
__constant__ static GMatf  _d_depth;
__constant__ static GMatf  _d_rigidnesses;
__constant__ static GMatf  _d_rigidnesses_sum;

__constant__ static GMatf2 _d_p2_map;
__constant__ static GMatf3 _d_p3_map;
__constant__ static GMatf3 _d_trifocal_0_map;
__constant__ static GMatf3 _d_trifocal_1_map;
__constant__ static GMatf3 _d_trifocal_2_map;
__constant__ static GMatf  _d_trifocal_squared_error;

static GMatf2 d_flows_1;
static GMatf2 d_flows_2;
static GMatf  d_disparities;
static GMatf  d_depth;
static GMatf  d_rigidnesses;
static GMatf  d_rigidnesses_sum;

static GMatf2 d_p2_map;
static GMatf3 d_p3_map;
static GMatf3 d_trifocal_0_map;
static GMatf3 d_trifocal_1_map;
static GMatf3 d_trifocal_2_map;
static GMatf  d_trifocal_squared_error;

__device__ __inline__ static void project_p2_to_p3(float px, float py, float depth, float& ox, float& oy, float& oz)
{
	ox = (_K4_inv[0] * px + _K4_inv[1]) * depth;
	oy = (_K4_inv[2] * py + _K4_inv[3]) * depth;
	oz = depth;
}

__device__ __inline__ static void project_p3_to_p2(float ox, float oy, float oz, float& px, float& py)
{
	px = (_K4[0] * ox + _K4[1] * oz) / oz;
	py = (_K4[2] * oy + _K4[3] * oz) / oz;
}

__device__ __inline__ static void transform_p3_across_frame(float& ox, float& oy, float& oz, int f)
{
	float ox_temp = ox * _Rs[f][0][0] + oy * _Rs[f][0][1] + oz * _Rs[f][0][2];
	float oy_temp = ox * _Rs[f][1][0] + oy * _Rs[f][1][1] + oz * _Rs[f][1][2];
	float oz_temp = ox * _Rs[f][2][0] + oy * _Rs[f][2][1] + oz * _Rs[f][2][2];
	ox = ox_temp + _ts[f][0];
	oy = oy_temp + _ts[f][1];
	oz = oz_temp + _ts[f][2];
}

__global__ static void compute_rigidnesses_sum()
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if ((x >= _w) || (y >= _h)) { return; }
	_d_rigidnesses_sum.at(x, y) = 0;
	for (int i = 0; i < _N; i++) { _d_rigidnesses_sum.at(x, y) += _d_rigidnesses.at(x, y, i); }
}

__global__ static void compute_p3p_map(int active_index, float rigidness_threshold, float rigidness_sum_threshold, float sample_min_depth, float sample_max_depth, int max_trace_on_flow, bool disparities_enable, bool disparities_use_0, bool trifocal_enable, bool trifocal_enable_flow_2, bool tf_use_flow_2, float trifocal_squared_error_threshold)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if ((x >= _w) || (y >= _h)) { return; }

	_d_p2_map.at(x, y) = make_float2(CUDART_NAN_F, CUDART_NAN_F);
	_d_p3_map.at(x, y) = make_float3(CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F);

	_d_trifocal_0_map.at(x, y) = make_float3(CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F);
	_d_trifocal_1_map.at(x, y) = make_float3(CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F);
	_d_trifocal_2_map.at(x, y) = make_float3(CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F);

	_d_trifocal_squared_error.at(x, y) = CUDART_NAN_F;

	float depth = _d_depth.at(x, y);
	if ((depth < sample_min_depth) || ((sample_max_depth > 0) && (depth > sample_max_depth))) { return; }

	// TODO: ?
	if ((_d_rigidnesses_sum.at(x, y) < rigidness_sum_threshold) && (rigidness_sum_threshold > (_N + 1))) { return; }

	int n_trace_on_flow = 0;
	int trace_idx = (max_trace_on_flow > 0) ? max(0, active_index - max_trace_on_flow + 1) : 0;
	float trace_product = 1;

	for (int i = active_index; i >= trace_idx; --i)
	{
	trace_product *= _d_rigidnesses.at(x, y, i); // TODO: ?
	if (trace_product <= rigidness_threshold) { break; }
	n_trace_on_flow++;
	}

	if (n_trace_on_flow <= 0) { return;	}

	int trace_start = active_index - n_trace_on_flow + 1;

	float ox;
	float oy;
	float oz;

	project_p2_to_p3(x, y, depth, ox, oy, oz);

	for (int i = 0; i < trace_start; ++i) { transform_p3_across_frame(ox, oy, oz, i); }

	float px;
	float py;

	project_p3_to_p2(ox, oy, oz, px, py);

	for (int i = trace_start; i <= active_index; ++i)
	{
	if ((px <= 0) || (px >= _w) || (py <= 0) || (py >= _h)) { return; }
	float2 dp = _d_flows_1.at_tex(px, py, i);
	px += dp.x;
	py += dp.y;
	}

	for (int i = trace_start; i < active_index; ++i) { transform_p3_across_frame(ox, oy, oz, i); }

	if ((oz <= sample_min_depth) || ((sample_max_depth > 0) && (oz >= sample_max_depth))) { return; }

	_d_p2_map.at(x, y) = make_float2(px, py);
	_d_p3_map.at(x, y) = make_float3(ox, oy, oz);

	// Extension

	float wx;
	float wy;

	project_p3_to_p2(ox, oy, oz, wx, wy);

	if ((wx <= 0) || (wx >= _w) || (wy <= 0) || (wy >= _h)) { return; }
	if ((px <= 0) || (px >= _w) || (py <= 0) || (py >= _h)) { return; }

	bool disparities_copy_0 = disparities_enable && disparities_use_0;

	_d_trifocal_0_map.at(x, y) = make_float3(wx, wy, disparities_copy_0 ? _d_disparities.at_tex(wx, wy, active_index + 0) : oz);
	_d_trifocal_1_map.at(x, y) = make_float3(px, py, disparities_enable ? _d_disparities.at_tex(px, py, active_index + 1) : 0.0f);

	if (!trifocal_enable) { return; }

	float2 d21 = _d_flows_1.at_tex(px, py, active_index + 1);
	float2 p2a = make_float2(px + d21.x, py + d21.y);

	float2 p2;
	float squared_error;

	if (trifocal_enable_flow_2)
	{
	float2 d22 = _d_flows_2.at_tex(wx, wy, active_index + 0);
	float2 p2b = make_float2(wx + d22.x, wy + d22.y);

	p2 = tf_use_flow_2 ? p2b : p2a;

	float dx = p2a.x - p2b.x;
	float dy = p2a.y - p2b.y;
	
	squared_error = (dx * dx) + (dy * dy);
	if ((trifocal_squared_error_threshold > 0) && (squared_error >= trifocal_squared_error_threshold)) { return; }
	}
	else
	{
	p2 = p2a;
	squared_error = 0;
	}

	float qx = p2.x;
	float qy = p2.y;

	if ((qx <= 0) || (qx >= _w) || (qy <= 0) || (qy >= _h)) { return; }

	_d_trifocal_2_map.at(x, y) = make_float3(qx, qy, disparities_enable ? _d_disparities.at_tex(qx, qy, active_index + 2) : 0.0f);

	_d_trifocal_squared_error.at(x, y) = squared_error;
}

int
collect_p3p_instances
(
	float const* h_flows_1[],
	float const* h_flows_2[],
	float const* h_disparities[],

	float const* h_rigidnesses[],
	float const* h_depth,
	float const* h_K,
	float const* h_Rs[],
	float const* h_ts[],

	int N, // n_flows
	int w,
	int h,
	int active_index,
	float rigidness_threshold,
	float rigidness_sum_threshold,
	float sample_min_depth,
	float sample_max_depth,
	int max_trace_on_flow,
	int disparities_enable,
	int disparities_use_0,
	int trifocal_enable,
	int trifocal_enable_flow_2,
	int trifocal_index_2,
	float trifocal_squared_error_threshold,

	float* h_o_p3_map,
	float* h_o_p2_map,
	float* h_o_trifocal_0_map,
	float* h_o_trifocal_1_map,
	float* h_o_trifocal_2_map,
	float* h_o_trifocal_squared_error
	
)
{
	// for pixel-wise op
	const dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH);
	const dim3 grid_size(DIV_CEIL(w, BLOCK_WIDTH), DIV_CEIL(h, BLOCK_WIDTH));

	// copy params to constant memory
	static float cache_symbols[3] = { 0 };
	CUDA_UPDATE_SYMBOL_IF_CHANGED(N, cache_symbols[0], _N);
	CUDA_UPDATE_SYMBOL_IF_CHANGED(w, cache_symbols[1], _w);
	CUDA_UPDATE_SYMBOL_IF_CHANGED(h, cache_symbols[2], _h);


	// copy camera info to constant memory
	// b_xxx stands for temp buffer
	if (h_K) {
		float b_K4[4]{ h_K[0] , h_K[2], h_K[4],h_K[5] };
		float b_K4_inv[4]{ 1.f / h_K[0], -h_K[2] / h_K[0], 1.f / h_K[4], -h_K[5] / h_K[4] };
		cudaMemcpyToSymbol(_K4, b_K4, 4 * sizeof(float));
		cudaMemcpyToSymbol(_K4_inv, b_K4_inv, 4 * sizeof(float));
	}


	if (h_Rs) {
		float b_R[MAX_FRAMES][3][3];
		for (int f = 0; f < N; f++)
			memcpy(b_R[f], h_Rs[f], 9 * sizeof(float));
		cudaMemcpyToSymbol(_Rs, b_R, N * 9 * sizeof(float));
		gpuErrchk;
	}

	if (h_ts) {
		float b_t[MAX_FRAMES][3];
		for (int f = 0; f < N; f++)
			memcpy(b_t[f], h_ts[f], 3 * sizeof(float));
		cudaMemcpyToSymbol(_ts, b_t, N * 3 * sizeof(float));
		gpuErrchk;
	}


	int Nf = trifocal_enable ? (N + 1) : N;

	// copy flow to device
	if (d_flows_1.create(w, h, Nf, true)) {
		d_flows_1.bind_tex();
		cudaMemcpyToSymbol(_d_flows_1, &d_flows_1, sizeof(GMatf2));
	}
	if (h_flows_1) {	
		for (int f = 0; f < Nf; f++)
			d_flows_1.copy_from_host((float2*)h_flows_1[f], make_cudaPos(0, 0, f), w, h, 1);
		gpuErrchk;
	}

	if (d_flows_2.create(w, h, Nf-1, true))
	{
		d_flows_2.bind_tex();
		cudaMemcpyToSymbol(_d_flows_2, &d_flows_2, sizeof(GMatf2));
	}
	if (h_flows_2)
	{
		for (int f = 0; f < Nf-1; ++f)
		{
			d_flows_2.copy_from_host((float2*)h_flows_2[f], make_cudaPos(0, 0, f), w, h, 1);
		}
		gpuErrchk;
	}

	if (d_disparities.create(w, h, Nf + 1, true))
	{
		d_disparities.bind_tex();
		cudaMemcpyToSymbol(_d_disparities, &d_disparities, sizeof(GMatf));
	}
	if (h_disparities)
	{
		for (int f = 0; f < Nf + 1; ++f)
		{
			d_disparities.copy_from_host((float*)h_disparities[f], make_cudaPos(0, 0, f), w, h, 1);
		}
		gpuErrchk;
	}




	if (d_rigidnesses_sum.create(w, h, 1))
		cudaMemcpyToSymbol(_d_rigidnesses_sum, &d_rigidnesses_sum, sizeof(GMatf));
	gpuErrchk;

	// copy rigidnesses
	if (d_rigidnesses.create(w, h, N, true))
		cudaMemcpyToSymbol(_d_rigidnesses, &d_rigidnesses, sizeof(GMatf));
	if (h_rigidnesses) {
		for (int f = 0; f < N; f++)
			d_rigidnesses.copy_from_host(h_rigidnesses[f], make_cudaPos(0, 0, f), w, h, 1);
		compute_rigidnesses_sum << <grid_size, block_size >> > ();
	}
	gpuErrchk;

	// copy depth to device
	if (d_depth.create(w, h, 1))
		cudaMemcpyToSymbol(_d_depth, &d_depth, sizeof(GMatf));
	if (h_depth) {
		d_depth.copy_from_host(h_depth, make_cudaPos(0, 0, 0), w, h, 1);
	}
	gpuErrchk;






	if (d_p2_map.create(w, h, 1)) { cudaMemcpyToSymbol(_d_p2_map, &d_p2_map, sizeof(d_p2_map)); }
	gpuErrchk;

	if (d_p3_map.create(w, h, 1)) { cudaMemcpyToSymbol(_d_p3_map, &d_p3_map, sizeof(d_p3_map)); }
	gpuErrchk;

	if (d_trifocal_0_map.create(w, h, 1)) { cudaMemcpyToSymbol(_d_trifocal_0_map, &d_trifocal_0_map, sizeof(d_trifocal_0_map)); }
	gpuErrchk;

	if (d_trifocal_1_map.create(w, h, 1)) { cudaMemcpyToSymbol(_d_trifocal_1_map, &d_trifocal_1_map, sizeof(d_trifocal_1_map)); }
	gpuErrchk;

	if (d_trifocal_2_map.create(w, h, 1)) { cudaMemcpyToSymbol(_d_trifocal_2_map, &d_trifocal_2_map, sizeof(d_trifocal_2_map)); }
	gpuErrchk;

	if (d_trifocal_squared_error.create(w, h, 1)) { cudaMemcpyToSymbol(_d_trifocal_squared_error, &d_trifocal_squared_error, sizeof(d_trifocal_squared_error)); }
	gpuErrchk;

	compute_p3p_map << <grid_size, block_size >> > (active_index, rigidness_threshold, rigidness_sum_threshold, sample_min_depth, sample_max_depth, max_trace_on_flow, disparities_enable, disparities_use_0, trifocal_enable, trifocal_enable_flow_2, trifocal_index_2, trifocal_squared_error_threshold);
	gpuErrchk;

	if (h_o_p2_map) { d_p2_map.copy_to_host((float2*)h_o_p2_map, make_cudaPos(0, 0, 0), w, h, 1); }
	if (h_o_p3_map) { d_p3_map.copy_to_host((float3*)h_o_p3_map, make_cudaPos(0, 0, 0), w, h, 1); }
	if (h_o_trifocal_0_map) { d_trifocal_0_map.copy_to_host((float3*)h_o_trifocal_0_map, make_cudaPos(0, 0, 0), w, h, 1); }
	if (h_o_trifocal_1_map) { d_trifocal_1_map.copy_to_host((float3*)h_o_trifocal_1_map, make_cudaPos(0, 0, 0), w, h, 1); }
	if (h_o_trifocal_2_map) { d_trifocal_2_map.copy_to_host((float3*)h_o_trifocal_2_map, make_cudaPos(0, 0, 0), w, h, 1); }
	if (h_o_trifocal_squared_error) { d_trifocal_squared_error.copy_to_host((float*)h_o_trifocal_squared_error, make_cudaPos(0, 0, 0), w, h, 1); }

	gpuErrchk;
	
	return cudaSuccess;
}
