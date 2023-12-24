#pragma once

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../N_body_config/N_body_cfg.hpp"
#include "../struct_Body.hpp"
#include "../struct_ode_system.hpp"
#include "../utils/norm_difference.hpp"


template <typename T>
void ode_RK2_MPI(ode_system<T>& sys, std::size_t& iter, T& timer,
				 MPI_Datatype type, int* counts, int* offsets, std::size_t id,
				 const std::string path = "", const T& eps = 1e-6) {
	/**
	 * Функция частичного умножение вектора на коэффициет alpha
	 */
	auto mult = [](std::vector<Body<T>>& vec, const T alpha, std::size_t begin,
				   std::size_t end) {
		for (std::size_t i = begin; i < end; ++i) {
			vec[i].r *= alpha;
			vec[i].v *= alpha;
		}
	};

	/**
	 * Функция частичного сложения векторов, содержащих тела
	 */
	auto add = [](std::vector<Body<T>>& vec1, const std::vector<Body<T>>& vec2,
				  std::size_t begin, std::size_t end) {
		for (std::size_t i = begin; i < end; ++i) {
			vec1[i].r += vec2[i].r;
			vec1[i].v += vec2[i].v;
		}
	};

	std::ofstream fout;

	if (!path.empty() && id == 0) fout = std::ofstream(path);

	if (!path.empty() && id == 0 && !fout.is_open()) {
		std::cout << "Could not open file for writing" << std::endl;
	} else {
		fout << "time;r11;r12;r13;r21;r22;r23;r31;r32;r33;r41;r42;r43\n";
	}

	std::size_t N_of_bodies = sys.traj_init.size();
	std::vector<Body<T>> sol(sys.traj_init), k(N_of_bodies), tmp(N_of_bodies);

	std::size_t begin = offsets[id];
	std::size_t count = counts[id];
	std::size_t end = begin + count;


	if (!path.empty() && id == 0) write_bodies_coords(fout, sys.t_start, sol);

	T t = sys.t_start + sys.tau;
	timer = -MPI_Wtime();


	for (size_t i = 0; i < sys.traj_init.size(); ++i)
		tmp[i].m = sys.traj_init[i].m;

	while (true) {
		sys.N_body_func(sol, k, begin, end);
		mult(k, sys.tau / 2, begin, end);

		MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, k.data(), counts,
					   offsets, type, MPI_COMM_WORLD);

		std::copy_n(sol.begin() + begin, count, tmp.begin() + begin);

		add(tmp, k, begin, end);

		MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, tmp.data(), counts,
					   offsets, type, MPI_COMM_WORLD);

		sys.N_body_func(tmp, k, begin, end);

		mult(k, sys.tau, begin, end);
		add(sol, k, begin, end);

		MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, sol.data(), counts,
					   offsets, type, MPI_COMM_WORLD);

		if (!path.empty() && id == 0) {
			write_bodies_coords(fout, t, sol);
		}

		if (t > sys.t_final) break;
		t += sys.tau;
	}

	timer += MPI_Wtime();

	fout.close();

	return;
}
