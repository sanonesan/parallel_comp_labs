#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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


typedef float T;
typedef float4 T3;
const std::string T_TYPE_NAME = "float";
#define EPS 1e-6f

// typedef double T;
// typedef double4 T3;
// const std::string T_TYPE_NAME = "double";
// #define EPS 1e-6

#define BLOCK_SIZE 64
#define G T(6.67e-11)


void cudaCheckError(cudaError_t error) {
	if (error != cudaSuccess)
		printf("Error %d: %s\n", error, cudaGetErrorString(error));
	// printf("Error %d: %s (file %s, line %d)\n", error,
	// cudaGetErrorString(error), __FILE__, __LINE__);
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

/**
 * Functon for writing trajectories of bodies into file
 */
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


/**
 * BEGIN BLOCK
 * Kernel functions for Mult and add
 */


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

__global__ void mult_n_add(T3* vec1, T alpha, T3* vec2, T3* vec_res, int size) {
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size) {
		vec_res[i].x = vec1[i].x + alpha * vec2[i].x;
		vec_res[i].y = vec1[i].y + alpha * vec2[i].y;
		vec_res[i].z = vec1[i].z + alpha * vec2[i].z;
	}
}

/**
 * END BLOCK
 * Kernel functions for Mult and add
 */

__global__ void calc_acceleration(T* m, T3* r, T3* v, int size) {
	unsigned int globIdx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int locIdx = threadIdx.x;

	T3 r_glob_tmp;
	if (globIdx < size) {
		r_glob_tmp = r[globIdx];
	} else {
		return;
	}

	__shared__ T m_shared[BLOCK_SIZE];
	__shared__ T3 r_shared[BLOCK_SIZE];


	T3 ri_rj;
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
				ri_rj.x = r_glob_tmp.x - r_shared[j].x;
				ri_rj.y = r_glob_tmp.y - r_shared[j].y;
				ri_rj.z = r_glob_tmp.z - r_shared[j].z;

				den = ri_rj.x * ri_rj.x + ri_rj.y * ri_rj.y + ri_rj.z * ri_rj.z;

				// if T == float

				den = sqrtf(den);
				den = den * den * den;
				den = __fdividef(m_shared[j], fmax(den, EPS * EPS * EPS));

				// if T == double

				// den = sqrt(den);
				// den = den * den * den;
				// den = m_shared[j] / fmax(den, EPS * EPS * EPS);

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


// RHS function for Nbody problem
void Nbody_func_async(cudaStream_t CUDA_STREAM, T* m, T3* r, T3* v, T3* r_res,
					  T3* v_res, dim3 threadsPerBlock, dim3 blocksInGrid,
					  int size) {
	cudaMemcpyAsync(r_res, v, sizeof(T3) * size, cudaMemcpyDeviceToDevice,
					CUDA_STREAM);
	calc_acceleration<<<blocksInGrid, threadsPerBlock, 0, CUDA_STREAM>>>(
		m, r, v_res, size);
}

void ode_RK2_CUDA(const std::vector<T>& m, const std::vector<T3>& r,
				  const std::vector<T3>& v, T t_start, T t_final, T tau,
				  std::string path = "", std::string path_1 = "") {
	std::size_t size = m.size();

	// output info about algorithm execution
	std::ofstream fout;
	// output for 4body test
	std::ofstream fout_1;


	/*
	 * Output for collecting info about calculations
	 */
	if (!path.empty()) fout = std::ofstream(path);
	if (!path.empty() && !fout.is_open()) {
		std::cout << "Could not open file for writing" << std::endl;
	} else {
		fout << "type;n_bodies;n_steps;block_size;time_taken;avg_exec_time\n";
	}

	/**
	 * Output for 4Body test
	 */
	if (!path_1.empty()) fout_1 = std::ofstream(path_1);
	if (!path_1.empty() && !fout_1.is_open()) {
		std::cout << "Could not open file for writing" << std::endl;
	} else {
		fout_1 << "time;";
		// r11;r12;r13;r21;r22;r23;r31;r32;r33;\n";
		for (int i = 1; i < size; ++i) {
			for (auto j : {1, 2, 3}) {
				fout_1 << "r_" + std::to_string(i) + "_" + std::to_string(j) +
							  ";";
			}
		}
		fout_1 << "r_" + std::to_string(size) + "_" + std::to_string(1) + ";";
		fout_1 << "r_" + std::to_string(size) + "_" + std::to_string(2) + ";";
		fout_1 << "r_" + std::to_string(size) + "_" + std::to_string(3) + "\n";
	}

	dim3 threadsPerBlock = BLOCK_SIZE;
	dim3 blocksInGrid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// tmp vector of T3 type for 4body test output
	std::vector<T3> tmp;
	tmp.resize(size);

	// Creating CUDA stream for acync cudaMemcpy (cudaMemcpyAsync)
	cudaStream_t CUDA_STREAM;
	cudaStreamCreate(&CUDA_STREAM);

	// Masses, Radii and Velocities strored in the GPU memory
	T* m_cuda;
	T3 *r_cuda, *v_cuda;
	cudaCheckError(cudaMalloc(&m_cuda, sizeof(T) * size));
	cudaCheckError(cudaMalloc(&r_cuda, sizeof(T3) * size));
	cudaCheckError(cudaMalloc(&v_cuda, sizeof(T3) * size));


	// Coefficient for RK2 algorithm
	T3 *k_r_cuda, *k_v_cuda;

	cudaCheckError(cudaMalloc(&k_r_cuda, sizeof(T3) * size));
	cudaCheckError(cudaMalloc(&k_v_cuda, sizeof(T3) * size));

	// Temporary Masses, Radii and Velocities strored in the GPU memory
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


	// Set time for the first step of RK2
	T t = t_start + tau;
	// Setting timer to measure time of calculations
	cudaEvent_t timer_start, timer_stop;
	cudaCheckError(cudaEventCreate(&timer_start));
	cudaCheckError(cudaEventCreate(&timer_stop));
	// Start timer
	cudaEventRecord(timer_start);
	while (true) {
		/**
		 * RK2 algorithm
		 */
		Nbody_func_async(CUDA_STREAM, m_cuda, r_cuda, v_cuda, k_r_cuda,
						 k_v_cuda, threadsPerBlock, blocksInGrid, size);
		mult_n_add<<<blocksInGrid, threadsPerBlock, 0, CUDA_STREAM>>>(
			r_cuda, tau / 2, k_r_cuda, r_tmp_cuda, size);
		mult_n_add<<<blocksInGrid, threadsPerBlock, 0, CUDA_STREAM>>>(
			v_cuda, tau / 2, k_v_cuda, v_tmp_cuda, size);

		Nbody_func_async(CUDA_STREAM, m_cuda, r_tmp_cuda, v_tmp_cuda, k_r_cuda,
						 k_v_cuda, threadsPerBlock, blocksInGrid, size);

		mult_n_add<<<blocksInGrid, threadsPerBlock, 0, CUDA_STREAM>>>(
			r_cuda, tau, k_r_cuda, r_cuda, size);
		mult_n_add<<<blocksInGrid, threadsPerBlock, 0, CUDA_STREAM>>>(
			v_cuda, tau, k_v_cuda, v_cuda, size);

		/**
		 * Commented code is another implementation of RK2 algorithm,
		 * where mult and add functions aren't gathered
		 */


		// Nbody_func_async(CUDA_STREAM, m_cuda, r_cuda, v_cuda, k_r_cuda,
		// 				 k_v_cuda, threadsPerBlock, blocksInGrid, size);
		//
		// mult<<<blocksInGrid, threadsPerBlock, 0, CUDA_STREAM>>>(k_r_cuda,
		// 														tau / 2, size);
		// mult<<<blocksInGrid, threadsPerBlock, 0, CUDA_STREAM>>>(k_v_cuda,
		// 														tau / 2, size);
		//
		// add<<<blocksInGrid, threadsPerBlock, 0, CUDA_STREAM>>>(
		// 	r_cuda, k_r_cuda, r_tmp_cuda, size);
		// add<<<blocksInGrid, threadsPerBlock, 0, CUDA_STREAM>>>(
		// 	v_cuda, k_v_cuda, v_tmp_cuda, size);
		//
		//
		// Nbody_func_async(CUDA_STREAM, m_cuda, r_tmp_cuda, v_tmp_cuda,
		// k_r_cuda, 				 k_v_cuda, threadsPerBlock, blocksInGrid,
		// size);
		//
		//
		// mult<<<blocksInGrid, threadsPerBlock, 0, CUDA_STREAM>>>(k_r_cuda,
		// tau, 														size);
		// mult<<<blocksInGrid, threadsPerBlock, 0, CUDA_STREAM>>>(k_v_cuda,
		// tau, 														size);
		// add<<<blocksInGrid, threadsPerBlock, 0, CUDA_STREAM>>>(r_cuda,
		// k_r_cuda, size); add<<<blocksInGrid, threadsPerBlock, 0,
		// CUDA_STREAM>>>(v_cuda,
		// k_v_cuda, size);

		if (!path_1.empty()) {
			/**
			 * Output for 4body test
			 */
			cudaMemcpyAsync(tmp.data(), r_cuda, sizeof(T3) * size,
							cudaMemcpyDeviceToHost, CUDA_STREAM);

			/**
			 * If output errors occured try this code instead of Acync.
			 * It should be slower, but it definitely works.
			 */
			// cudaDeviceSynchronize();
			// cudaMemcpy(tmp.data(), r_cuda, sizeof(T3) * size,
			// 				cudaMemcpyDeviceToHost);
			write_bodies_coords(fout_1, t, tmp.data(), size);
		}

		if (t > t_final) break;
		t += tau;
	}
	// Stop timer
	cudaEventRecord(timer_stop);
	cudaEventSynchronize(timer_stop);

	// Calculate time taken for algorithm
	float elapsed_time = 0.;
	cudaEventElapsedTime(&elapsed_time, timer_start, timer_stop);
	cudaEventDestroy(timer_start);
	cudaEventDestroy(timer_stop);
	// Calculate average time for on step of algorithm
	double avg_exec_time = elapsed_time / ((t_final - t_start) / tau);

	// Destroying variables stored in the GPU
	cudaStreamDestroy(CUDA_STREAM);
	cudaCheckError(cudaFree(m_cuda));
	cudaCheckError(cudaFree(r_cuda));
	cudaCheckError(cudaFree(v_cuda));

	cudaCheckError(cudaFree(k_r_cuda));
	cudaCheckError(cudaFree(k_v_cuda));

	cudaCheckError(cudaFree(r_tmp_cuda));
	cudaCheckError(cudaFree(v_tmp_cuda));

	// Output collected information of algorithm execution
	fout << T_TYPE_NAME + ";";
	fout << size << ";" << (int)((t_final - t_start) / tau) << ";" << BLOCK_SIZE
		 << ";" << elapsed_time / 1000. << ";" << avg_exec_time / 1000. << "\n";

	// Do not forget for close files
	fout.close();
	fout_1.close();

	return;
}

void test_4body() {
	std::vector<T> m;
	std::vector<T3> r, v;

	std::string read_path;
	std::string out_path;

	// testing on Personal Comuter

	// read_path = "./include/4body.txt";
	// read_data(read_path, m, r, v);
	//
	// out_path = "../output/res_4body_" + T_TYPE_NAME + "_" +
	// 		   std::to_string(BLOCK_SIZE) + "_.csv";
	//
	// ode_RK2_CUDA(m, r, v, 0., 20., 0.1, out_path,
	// 			 "../output/res4body_async.csv");

	// testing on claster

	read_path = config_lab_5_CUDA::PATH_src_folder + "/include/4body.txt";
	read_data(read_path, m, r, v);

	out_path = config_lab_5_CUDA::PATH_output_folder + "/res_4body_" +
			   T_TYPE_NAME + "_" + std::to_string(BLOCK_SIZE) + "_.csv";

	ode_RK2_CUDA(m, r, v, 0., 20., 0.1, out_path,
				 config_lab_5_CUDA::PATH_output_folder + "/res4body_async.csv");
}

void test_random_bodies() {
	std::vector<T> m;
	std::vector<T3> r, v;

	std::string read_path;
	std::string out_path;


	// testing on Personal Comuter

	// read_path = "./include/10k_bodies.txt";
	// read_data(read_path, m, r, v);
	//
	// out_path = "../output/res_10k_" + T_TYPE_NAME + "_" +
	// 		   std::to_string(BLOCK_SIZE) + "_.csv";

	// testing on claster

	read_path = config_lab_5_CUDA::PATH_src_folder + "/include/10k_bodies.txt";
	read_data(read_path, m, r, v);

	out_path = config_lab_5_CUDA::PATH_output_folder + "/res_10k_" +
			   T_TYPE_NAME + "_" + std::to_string(BLOCK_SIZE) + "_.csv";


	ode_RK2_CUDA(m, r, v, 0., 2., 0.1, out_path);


	read_path = config_lab_5_CUDA::PATH_src_folder + "/include/50k_bodies.txt";
	read_data(read_path, m, r, v);

	out_path = config_lab_5_CUDA::PATH_output_folder + "/res_50k_" +
			   T_TYPE_NAME + "_" + std::to_string(BLOCK_SIZE) + "_.csv";

	ode_RK2_CUDA(m, r, v, 0., 2., 0.1, out_path);


	read_path = config_lab_5_CUDA::PATH_src_folder + "/include/100k_bodies.txt";
	read_data(read_path, m, r, v);

	out_path = config_lab_5_CUDA::PATH_output_folder + "/res_100k_" +
			   T_TYPE_NAME + "_" + std::to_string(BLOCK_SIZE) + "_.csv";

	ode_RK2_CUDA(m, r, v, 0., 2., 0.1, out_path);


	read_path = config_lab_5_CUDA::PATH_src_folder + "/include/500k_bodies.txt";
	read_data(read_path, m, r, v);

	out_path = config_lab_5_CUDA::PATH_output_folder + "/res_500k_" +
			   T_TYPE_NAME + "_" + std::to_string(BLOCK_SIZE) + "_.csv";

	ode_RK2_CUDA(m, r, v, 0., 2., 0.1, out_path);
}

int main(int argc, char** argv) {
	test_4body();
	// test_random_bodies();
	return 0;
}
