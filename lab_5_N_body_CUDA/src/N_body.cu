#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// Other
#include <stdio.h>

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

#include "config.h"

#define FLOAT__

#ifdef FLOAT__
typedef float T;
typedef float4 T3;

#else

typedef double T;
typedef double4 T3;

#endif

#define EPS T(1e-6)
#define BLOCK_SIZE 512
#define G T(6.67e-11)


void cudaCheckError(cudaError_t error) {
	if (error != cudaSuccess)
		printf("Error %d: %s\n", error, cudaGetErrorString(error));
	// printf("Error %d: %s (file %s, line %d)\n", error,
	// cudaGetErrorString(error), __FILE__, __LINE__);
}

T rand_num_type_T(T a, T b) {
	srand(time(NULL));
	return (b - a) * rand() / RAND_MAX + a;
}

void generate_data(std::size_t size, std::vector<T>& m, std::vector<T3>& r,
				   std::vector<T3>& v) {
	m.resize(size);
	r.resize(size);
	v.resize(size);

	for (std::size_t i = 0; i < size; ++i) {
		m[i] = rand_num_type_T(1e6, 1e8);

		r[i].x = rand_num_type_T(-1e2, 1e2);
		r[i].y = rand_num_type_T(-1e2, 1e2);
		r[i].z = rand_num_type_T(-1e2, 1e2);

		v[i].x = rand_num_type_T(-1e1, 1e1);
		v[i].y = rand_num_type_T(-1e1, 1e1);
		v[i].z = rand_num_type_T(-1e1, 1e1);
	}
}


void read_data(const std::string& path, std::vector<T>& m, std::vector<T3>& r,
			   std::vector<T3>& v) {
	std::ifstream fout(path);

	if (!fout.is_open()) {
		std::cout << "file is't open" << std::endl;
		throw 1;
	}

	std::size_t size = 0;
	fout >> size;

	m.resize(size);
	r.resize(size);
	v.resize(size);

	for (size_t i = 0; i < size; ++i) {
		fout >> m[i] >> r[i].x >> r[i].y >> r[i].z >> v[i].x >> v[i].y >>
			v[i].z;
	}

	fout.close();
}


void write_bodies_coords(std::ostream& fout, const T time, const T3* r,
						 const int size) {
	fout << std::scientific;
	fout << std::setprecision(16);
	fout << time << ";";
	for (size_t i = 0; i < size; ++i) {
		fout << r[i].x << ";" << r[i].y << ";" << r[i].z << ";";
	}
	fout << std::endl;
}

__global__ void mult(T3* vec, T alpha, int size) {
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size) {
		vec[i].x *= alpha;
		vec[i].y *= alpha;
		vec[i].z *= alpha;
	}
}
__global__ void mult(T3* vec, T alpha, T3* res, int size) {
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size) {
		res[i].x = alpha * vec[i].x;
		res[i].y = alpha * vec[i].y;
		res[i].z = alpha * vec[i].z;
	}
}

__global__ void add(T3* vec1, T3* vec2, T3* res, int size) {
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size) {
		res[i].x = vec1[i].x + vec2[i].x;
		res[i].y = vec1[i].y + vec2[i].y;
		res[i].z = vec1[i].z + vec2[i].z;
	}
}

__global__ void add(T3* vec1, T3* vec2, int size) {
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size) {
		vec1[i].x += vec2[i].x;
		vec1[i].y += vec2[i].y;
		vec1[i].z += vec2[i].z;
	}
}

__global__ void calc_acceleration(T* m, T3* r, T3* v, int size) {
	unsigned int globIdx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int locIdx = threadIdx.x;

	__shared__ T m_shared[BLOCK_SIZE];
	__shared__ T3 r_shared[BLOCK_SIZE];

	T3 ri_rj;

	T3 th = r[globIdx];
	T den;

	T3 a_tmp{
		0.,
		0.,
		0.,
	};
	for (int i = 0; i < size; i += BLOCK_SIZE) {
		m_shared[locIdx] = m[i + locIdx];
		r_shared[locIdx] = r[i + locIdx];

		__syncthreads();

		for (int j = 0; j < BLOCK_SIZE; ++j) {
			if (i + j < size) {
				ri_rj.x = th.x - r_shared[j].x;
				ri_rj.y = th.y - r_shared[j].y;
				ri_rj.z = th.z - r_shared[j].z;

				den = ri_rj.x * ri_rj.x + ri_rj.y * ri_rj.y + ri_rj.z * ri_rj.z;
#ifdef FLOAT__
				den *= sqrtf(den);
				den = __fdividef(m_shared[j], fmax(den, EPS));
#else
				den *= sqrt(den);
				den = m_shared[j] / fmax(den, EPS);
#endif

				a_tmp.x += ri_rj.x * den;
				a_tmp.y += ri_rj.y * den;
				a_tmp.z += ri_rj.z * den;
			}
		}

		__syncthreads();
	}
	if (globIdx < size) {
		v[globIdx].x = (-G) * a_tmp.x;
		v[globIdx].y = (-G) * a_tmp.y;
		v[globIdx].z = (-G) * a_tmp.z;
	}
}

__global__ void mult_and_add(T3* vec1, T3* vec2, T3* res, T alpha, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		res[i].x = vec1[i].x + alpha * vec2[i].x;
		res[i].y = vec1[i].y + alpha * vec2[i].y;
		res[i].z = vec1[i].z + alpha * vec2[i].z;
	}
}

void Nbody_func(T* m, T3* r, T3* v, T3* r_res, T3* v_res, dim3 threadsPerBlock,
				dim3 blocksInGrid, int size) {
	cudaMemcpy(r_res, v, sizeof(T3) * size, cudaMemcpyDeviceToDevice);
	calc_acceleration<<<blocksInGrid, threadsPerBlock>>>(m, r, v_res, size);
}

void ode_RK2_CUDA(const std::vector<T>& m, const std::vector<T3>& r,
				  const std::vector<T3>& v, T t_start, T t_final, T tau,
				  std::string path = "") {
	std::size_t size = m.size();

	std::ofstream fout;
	//
	if (!path.empty()) fout = std::ofstream(path);
	//
	if (!path.empty() && !fout.is_open()) {
		std::cout << "Could not open file for writing" << std::endl;
	} else {
		fout << "type;n_bodies;n_steps;block_size:time_taken;avg_exec_time\n";
	}


	dim3 threadsPerBlock = BLOCK_SIZE;
	dim3 blocksInGrid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;


	T* m_cuda;
	T3 *r_cuda, *v_cuda;
	cudaCheckError(cudaMalloc(&m_cuda, sizeof(T) * size));
	cudaCheckError(cudaMalloc(&r_cuda, sizeof(T3) * size));
	cudaCheckError(cudaMalloc(&v_cuda, sizeof(T3) * size));


	T3 *k1_r_cuda, *k2_r_cuda, *k1_v_cuda, *k2_v_cuda;

	cudaCheckError(cudaMalloc(&k1_r_cuda, sizeof(T3) * size));
	cudaCheckError(cudaMalloc(&k2_r_cuda, sizeof(T3) * size));
	cudaCheckError(cudaMalloc(&k1_v_cuda, sizeof(T3) * size));
	cudaCheckError(cudaMalloc(&k2_v_cuda, sizeof(T3) * size));

	T3 *r_tmp_cuda, *v_tmp_cuda;
	cudaCheckError(cudaMalloc(&r_tmp_cuda, sizeof(T3) * size));
	cudaCheckError(cudaMalloc(&v_tmp_cuda, sizeof(T3) * size));


	// Copy from host to device
	cudaCheckError(
		cudaMemcpy(m_cuda, m.data(), sizeof(T) * size, cudaMemcpyHostToDevice));

	cudaCheckError(cudaMemcpy(r_cuda, r.data(), sizeof(T3) * size,
							  cudaMemcpyHostToDevice));

	cudaCheckError(cudaMemcpy(v_cuda, v.data(), sizeof(T3) * size,
							  cudaMemcpyHostToDevice));


	T t = t_start + tau;
	cudaEvent_t timer_start, timer_stop;
	cudaCheckError(cudaEventCreate(&timer_start));
	cudaCheckError(cudaEventCreate(&timer_stop));

	cudaEventRecord(timer_start);

	int n_steps = 0;
	while (true) {
		++n_steps;

		Nbody_func(m_cuda, r_cuda, v_cuda, k1_r_cuda, k1_v_cuda,
				   threadsPerBlock, blocksInGrid, size);

		mult<<<blocksInGrid, threadsPerBlock>>>(k1_r_cuda, tau / 2, size);
		mult<<<blocksInGrid, threadsPerBlock>>>(k1_v_cuda, tau / 2, size);

		add<<<blocksInGrid, threadsPerBlock>>>(r_cuda, k1_r_cuda, r_tmp_cuda,
											   size);
		add<<<blocksInGrid, threadsPerBlock>>>(v_cuda, k1_v_cuda, v_tmp_cuda,
											   size);


		Nbody_func(m_cuda, r_tmp_cuda, v_tmp_cuda, k1_r_cuda, k1_v_cuda,
				   threadsPerBlock, blocksInGrid, size);


		mult<<<blocksInGrid, threadsPerBlock>>>(k1_r_cuda, tau, size);
		mult<<<blocksInGrid, threadsPerBlock>>>(k1_v_cuda, tau, size);

		add<<<blocksInGrid, threadsPerBlock>>>(r_cuda, k1_r_cuda, size);
		add<<<blocksInGrid, threadsPerBlock>>>(v_cuda, k1_v_cuda, size);

		if (t > t_final) break;
		t += tau;
	}

	cudaEventRecord(timer_stop);
	cudaEventSynchronize(timer_stop);

	float elapsed_time = 0.;
	cudaEventElapsedTime(&elapsed_time, timer_start, timer_stop);
	cudaEventDestroy(timer_start);
	cudaEventDestroy(timer_stop);
	double avg_exec_time = elapsed_time / ((t_final - t_start) / tau);


	cudaCheckError(cudaFree(m_cuda));
	cudaCheckError(cudaFree(r_cuda));
	cudaCheckError(cudaFree(v_cuda));

	cudaCheckError(cudaFree(k1_r_cuda));
	cudaCheckError(cudaFree(k1_v_cuda));
	cudaCheckError(cudaFree(k2_r_cuda));
	cudaCheckError(cudaFree(k2_v_cuda));

	cudaCheckError(cudaFree(r_tmp_cuda));
	cudaCheckError(cudaFree(v_tmp_cuda));

#ifdef FLOAT__
	fout << "float;";
#else
	fout << "double;";
#endif

	fout << size << ";" << (int)((t_final - t_start) / tau) << ";" << BLOCK_SIZE
		 << ";" << elapsed_time / 1000. << ";" << avg_exec_time / 1000. << "\n";

	fout.close();

	return;
}

void tests() {
	std::vector<T> m;
	std::vector<T3> r, v;

	std::string read_path;
	std::string out_path;

	read_path = config_lab_5_CUDA::PATH_src_folder + "/include/10k_bodies.txt";
	read_data(read_path, m, r, v);

	out_path = config_lab_5_CUDA::PATH_output_folder + "/output/res_10k_" +
			   std::to_string(BLOCK_SIZE) + "_.csv";

	ode_RK2_CUDA(m, r, v, 0., 2., 0.1, out_path);

	read_path = config_lab_5_CUDA::PATH_src_folder + "/include/50k_bodies.txt";
	read_data(read_path, m, r, v);

	out_path = config_lab_5_CUDA::PATH_output_folder + "/output/res_50k_" +
			   std::to_string(BLOCK_SIZE) + "_.csv";

	ode_RK2_CUDA(m, r, v, 0., 2., 0.1, out_path);

	read_path = config_lab_5_CUDA::PATH_src_folder + "/include/100k_bodies.txt";
	read_data(read_path, m, r, v);

	out_path = config_lab_5_CUDA::PATH_output_folder + "/output/res_100k_" +
			   std::to_string(BLOCK_SIZE) + "_.csv";

	ode_RK2_CUDA(m, r, v, 0., 2., 0.1, out_path);

	read_path = config_lab_5_CUDA::PATH_src_folder + "/include/500k_bodies.txt";
	read_data(read_path, m, r, v);

	out_path = config_lab_5_CUDA::PATH_output_folder + "/output/res_500k_" +
			   std::to_string(BLOCK_SIZE) + "_.csv";

	ode_RK2_CUDA(m, r, v, 0., 2., 0.1, out_path);
}

int main(int argc, char** argv) {
	tests();
	return 0;
}
