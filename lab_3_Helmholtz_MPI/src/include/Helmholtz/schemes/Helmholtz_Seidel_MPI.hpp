#pragma once

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "../Helmholtz_config/helmholtz_cfg.hpp"
#include "../class_Helmholtz.hpp"
#include "../helpers/divide_4_regions.hpp"
#include "../helpers/norm_difference.hpp"


template <typename T>
void Seidel_RB_MPI_Send_Recv(int id, int num_procs, Helmholtz_equation<T>& eq,
							 std::size_t& iter, const T& eps = 1e-6) {
	iter = 1;
	std::size_t n1 = eq.x2.size(), n2 = eq.x1.size();

	/**
	 * Rows for computations for processor
	 */
	std::vector<int> reg_sizes;
	std::vector<int> offsets;
	int reg_local_size, local_offset;

	divide_4_regions(eq, id, num_procs, reg_local_size, local_offset, reg_sizes,
					 offsets);
	std::vector<T> local_solution(reg_local_size * n2, 0.);
	std::vector<T> local_solution_prev(reg_local_size * n2, 0.);
	std::vector<T> row_above(n2, 0.), row_below(n2, 0.);

	T err, local_err;
	int source = id ? id - 1 : num_procs - 1;
	int dest = (id != num_procs - 1) ? id + 1 : 0;
	int send_count = (id != num_procs - 1) ? n2 : 0;
	int recv_count = id ? n2 : 0;

	/**
	 * Main body of Seidel_RB method
	 */
	eq.time = -MPI_Wtime();
	do {
		local_solution.swap(local_solution_prev);

		if (num_procs > 1) {
			MPI_Send(local_solution_prev.data() + (reg_local_size - 1) * n2,
					 send_count, MPI_DOUBLE, dest, 100, MPI_COMM_WORLD);
			MPI_Recv(row_above.data(), recv_count, MPI_DOUBLE, source, 100,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

			MPI_Send(local_solution_prev.data(), recv_count, MPI_DOUBLE, source,
					 200, MPI_COMM_WORLD);
			MPI_Recv(row_below.data(), send_count, MPI_DOUBLE, dest, 200,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}


		// Go through inner REDS
		for (std::size_t i = 1; i < reg_local_size - 1; ++i) {
			for (std::size_t j = 1 + (i + 1) % 2; j < n2 - 1; j += 2) {
				local_solution[i * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[i + local_offset], eq.x2[j]) +
					 local_solution_prev[i * n2 + j - 1] +
					 local_solution_prev[i * n2 + j + 1] +
					 local_solution_prev[(i - 1) * n2 + j] +
					 local_solution_prev[(i + 1) * n2 + j]) /
					eq._coef;
			}
		}

		// Go through upper rows REDS
		if (id != 0) {
			for (std::size_t j = 2; j < n2 - 1; j += 2) {
				local_solution[j] =
					(eq._h_2 * eq._f(eq.x1[local_offset], eq.x2[j]) +
					 local_solution_prev[j - 1] + local_solution_prev[j + 1] +
					 row_above[j] + local_solution_prev[n2 + j]) /
					eq._coef;
			}
		}


		// Go through lower rows REDS
		if (id != num_procs - 1) {
			for (std::size_t j = 1 + reg_local_size % 2; j < n2 - 1; j += 2) {
				local_solution[(reg_local_size - 1) * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[(reg_local_size - 1) + local_offset],
									 eq.x2[j]) +
					 local_solution_prev[(reg_local_size - 1) * n2 + j - 1] +
					 local_solution_prev[(reg_local_size - 1) * n2 + j + 1] +
					 local_solution_prev[(reg_local_size - 2) * n2 + j] +
					 row_below[j]) /
					eq._coef;
			}
		}

		if (num_procs > 1) {
			MPI_Send(local_solution.data() + (reg_local_size - 1) * n2,
					 send_count, MPI_DOUBLE, dest, 100, MPI_COMM_WORLD);
			MPI_Recv(row_above.data(), recv_count, MPI_DOUBLE, source, 100,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

			MPI_Send(local_solution.data(), recv_count, MPI_DOUBLE, source, 200,
					 MPI_COMM_WORLD);
			MPI_Recv(row_below.data(), send_count, MPI_DOUBLE, dest, 200,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}

		// Go through inner BLACKS
		for (std::size_t i = 1; i < reg_local_size - 1; ++i) {
			for (std::size_t j = 1 + i % 2; j < n2 - 1; j += 2) {
				local_solution[i * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[i + local_offset], eq.x2[j]) +
					 local_solution[i * n2 + j - 1] +
					 local_solution[i * n2 + j + 1] +
					 local_solution[(i - 1) * n2 + j] +
					 local_solution[(i + 1) * n2 + j]) /
					eq._coef;
			}
		}

		// Go through upper rows BLACKS
		if (id != 0) {
			for (std::size_t j = 1; j < n2 - 1; j += 2) {
				local_solution[j] =
					(eq._h_2 * eq._f(eq.x1[local_offset], eq.x2[j]) +
					 local_solution[j - 1] + local_solution[j + 1] +
					 row_above[j] + local_solution[n2 + j]) /
					eq._coef;
			}
		}


		// Go through lower rows BLACKS
		if (id != num_procs - 1) {
			for (std::size_t j = 1 + (reg_local_size - 1) % 2; j < n2 - 1;
				 j += 2) {
				local_solution[(reg_local_size - 1) * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[(reg_local_size - 1) + local_offset],
									 eq.x2[j]) +
					 local_solution[(reg_local_size - 1) * n2 + j - 1] +
					 local_solution[(reg_local_size - 1) * n2 + j + 1] +
					 local_solution[(reg_local_size - 2) * n2 + j] +
					 row_below[j]) /
					eq._coef;
			}
		}

		local_err =
			norm_difference(local_solution_prev, local_solution, reg_local_size,
							n2, config_Helmholtz_eq::NORM_TYPE, eps);

		MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		++iter;
	} while (sqrt(err) > eps);
	eq.time += MPI_Wtime();

	for (std::size_t i = 0; i < num_procs; ++i) {
		reg_sizes[i] *= n2;
		offsets[i] *= n2;
	}

	MPI_Gatherv(local_solution.data(), reg_local_size * n2, MPI_DOUBLE,
				eq.res.data(), reg_sizes.data(), offsets.data(), MPI_DOUBLE, 0,
				MPI_COMM_WORLD);

	return;
}


template <typename T>
void Seidel_RB_MPI_Sendrecv(int id, int num_procs, Helmholtz_equation<T>& eq,
							std::size_t& iter, const T& eps = 1e-6) {
	iter = 1;
	std::size_t n1 = eq.x2.size(), n2 = eq.x1.size();

	/**
	 * Rows for computations for processor
	 */
	std::vector<int> reg_sizes;
	std::vector<int> offsets;
	int reg_local_size, local_offset;

	divide_4_regions(eq, id, num_procs, reg_local_size, local_offset, reg_sizes,
					 offsets);
	std::vector<T> local_solution(reg_local_size * n2, 0.);
	std::vector<T> local_solution_prev(reg_local_size * n2, 0.);
	std::vector<T> row_above(n2, 0.), row_below(n2, 0.);

	T err, local_err;
	int source = id ? id - 1 : num_procs - 1;
	int dest = (id != num_procs - 1) ? id + 1 : 0;
	int send_count = (id != num_procs - 1) ? n2 : 0;
	int recv_count = id ? n2 : 0;

	/**
	 * Main body of Seidel_RB method
	 */
	eq.time = -MPI_Wtime();
	do {
		local_solution.swap(local_solution_prev);

		MPI_Sendrecv(local_solution_prev.data() + (reg_local_size - 1) * n2,
					 send_count, MPI_DOUBLE, dest, 100, row_above.data(),
					 recv_count, MPI_DOUBLE, source, 100, MPI_COMM_WORLD,
					 MPI_STATUSES_IGNORE);

		MPI_Sendrecv(local_solution_prev.data(), recv_count, MPI_DOUBLE, source,
					 200, row_below.data(), send_count, MPI_DOUBLE, dest, 200,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);


		// Go through inner REDS
		for (std::size_t i = 1; i < reg_local_size - 1; ++i) {
			for (std::size_t j = 1 + (i + 1) % 2; j < n2 - 1; j += 2) {
				local_solution[i * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[i + local_offset], eq.x2[j]) +
					 local_solution_prev[i * n2 + j - 1] +
					 local_solution_prev[i * n2 + j + 1] +
					 local_solution_prev[(i - 1) * n2 + j] +
					 local_solution_prev[(i + 1) * n2 + j]) /
					eq._coef;
			}
		}

		// Go through upper rows REDS
		if (id != 0) {
			for (std::size_t j = 2; j < n2 - 1; j += 2) {
				local_solution[j] =
					(eq._h_2 * eq._f(eq.x1[local_offset], eq.x2[j]) +
					 local_solution_prev[j - 1] + local_solution_prev[j + 1] +
					 row_above[j] + local_solution_prev[n2 + j]) /
					eq._coef;
			}
		}


		// Go through lower rows REDS
		if (id != num_procs - 1) {
			for (std::size_t j = 1 + reg_local_size % 2; j < n2 - 1; j += 2) {
				local_solution[(reg_local_size - 1) * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[(reg_local_size - 1) + local_offset],
									 eq.x2[j]) +
					 local_solution_prev[(reg_local_size - 1) * n2 + j - 1] +
					 local_solution_prev[(reg_local_size - 1) * n2 + j + 1] +
					 local_solution_prev[(reg_local_size - 2) * n2 + j] +
					 row_below[j]) /
					eq._coef;
			}
		}

		MPI_Sendrecv(local_solution.data() + (reg_local_size - 1) * n2,
					 send_count, MPI_DOUBLE, dest, 100, row_above.data(),
					 recv_count, MPI_DOUBLE, source, 100, MPI_COMM_WORLD,
					 MPI_STATUSES_IGNORE);

		MPI_Sendrecv(local_solution.data(), recv_count, MPI_DOUBLE, source, 200,
					 row_below.data(), send_count, MPI_DOUBLE, dest, 200,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);


		// Go through inner BLACKS
		for (std::size_t i = 1; i < reg_local_size - 1; ++i) {
			for (std::size_t j = 1 + i % 2; j < n2 - 1; j += 2) {
				local_solution[i * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[i + local_offset], eq.x2[j]) +
					 local_solution[i * n2 + j - 1] +
					 local_solution[i * n2 + j + 1] +
					 local_solution[(i - 1) * n2 + j] +
					 local_solution[(i + 1) * n2 + j]) /
					eq._coef;
			}
		}

		// Go through upper rows BLACKS
		if (id != 0) {
			for (std::size_t j = 1; j < n2 - 1; j += 2) {
				local_solution[j] =
					(eq._h_2 * eq._f(eq.x1[local_offset], eq.x2[j]) +
					 local_solution[j - 1] + local_solution[j + 1] +
					 row_above[j] + local_solution[n2 + j]) /
					eq._coef;
			}
		}


		// Go through lower rows BLACKS
		if (id != num_procs - 1) {
			for (std::size_t j = 1 + (reg_local_size - 1) % 2; j < n2 - 1;
				 j += 2) {
				local_solution[(reg_local_size - 1) * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[(reg_local_size - 1) + local_offset],
									 eq.x2[j]) +
					 local_solution[(reg_local_size - 1) * n2 + j - 1] +
					 local_solution[(reg_local_size - 1) * n2 + j + 1] +
					 local_solution[(reg_local_size - 2) * n2 + j] +
					 row_below[j]) /
					eq._coef;
			}
		}

		local_err =
			norm_difference(local_solution_prev, local_solution, reg_local_size,
							n2, config_Helmholtz_eq::NORM_TYPE, eps);

		MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		++iter;
	} while (sqrt(err) > eps);

	eq.time += MPI_Wtime();

	for (std::size_t i = 0; i < num_procs; ++i) {
		reg_sizes[i] *= n2;
		offsets[i] *= n2;
	}

	MPI_Gatherv(local_solution.data(), reg_local_size * n2, MPI_DOUBLE,
				eq.res.data(), reg_sizes.data(), offsets.data(), MPI_DOUBLE, 0,
				MPI_COMM_WORLD);

	return;
}


template <typename T>
void Seidel_RB_MPI_ISend_IRecv(int id, int num_procs, Helmholtz_equation<T>& eq,
							   std::size_t& iter, const T& eps = 1e-6) {
	iter = 1;
	std::size_t n1 = eq.x2.size(), n2 = eq.x1.size();

	MPI_Request* send_req_1;
	MPI_Request* send_req_2;
	MPI_Request* recv_req_1;
	MPI_Request* recv_req_2;

	/**
	 * Rows for computations for processor
	 */
	std::vector<int> reg_sizes;
	std::vector<int> offsets;
	int reg_local_size, local_offset;

	divide_4_regions(eq, id, num_procs, reg_local_size, local_offset, reg_sizes,
					 offsets);

	send_req_1 = new MPI_Request[2];
	send_req_2 = new MPI_Request[2];
	recv_req_1 = new MPI_Request[2];
	recv_req_2 = new MPI_Request[2];

	std::vector<T> local_solution(reg_local_size * n2, 0.);
	std::vector<T> local_solution_prev(reg_local_size * n2, 0.);
	std::vector<T> row_above(n2, 0.), row_below(n2, 0.);

	T err, local_err;

	int source = (id != 0) ? id - 1 : num_procs - 1;
	int dest = (id != num_procs - 1) ? id + 1 : 0;

	int send_count = (id != 0) ? n2 : 0;
	int recv_count = (id != num_procs - 1) ? n2 : 0;

	MPI_Send_init(local_solution_prev.data(), send_count, MPI_DOUBLE, source,
				  100, MPI_COMM_WORLD, send_req_1);
	MPI_Recv_init(row_below.data(), recv_count, MPI_DOUBLE, dest, MPI_ANY_TAG,
				  MPI_COMM_WORLD, recv_req_1);

	MPI_Send_init(local_solution_prev.data() + (reg_local_size - 1) * n2,
				  recv_count, MPI_DOUBLE, dest, 100, MPI_COMM_WORLD,
				  send_req_1 + 1);
	MPI_Recv_init(row_above.data(), send_count, MPI_DOUBLE, source, MPI_ANY_TAG,
				  MPI_COMM_WORLD, recv_req_1 + 1);

	MPI_Send_init(local_solution.data(), send_count, MPI_DOUBLE, source, 100,
				  MPI_COMM_WORLD, send_req_2);
	MPI_Recv_init(row_below.data(), recv_count, MPI_DOUBLE, dest, MPI_ANY_TAG,
				  MPI_COMM_WORLD, recv_req_2);

	MPI_Send_init(local_solution.data() + (reg_local_size - 1) * n2, recv_count,
				  MPI_DOUBLE, dest, 100, MPI_COMM_WORLD, send_req_2 + 1);
	MPI_Recv_init(row_above.data(), send_count, MPI_DOUBLE, source, MPI_ANY_TAG,
				  MPI_COMM_WORLD, recv_req_2 + 1);

	/**
	 * Main body of Seidel_RB method
	 */
	eq.time = -MPI_Wtime();
	do {
		local_solution.swap(local_solution_prev);

		if (iter % 2 == 0) {
			MPI_Startall(2, send_req_1);
			MPI_Startall(2, recv_req_1);
		} else {
			MPI_Startall(2, send_req_2);
			MPI_Startall(2, recv_req_2);
		}


		// Go through inner REDS
		for (std::size_t i = 1; i < reg_local_size - 1; ++i) {
			for (std::size_t j = 1 + (i + 1) % 2; j < n2 - 1; j += 2) {
				local_solution[i * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[i + local_offset], eq.x2[j]) +
					 local_solution_prev[i * n2 + j - 1] +
					 local_solution_prev[i * n2 + j + 1] +
					 local_solution_prev[(i - 1) * n2 + j] +
					 local_solution_prev[(i + 1) * n2 + j]) /
					eq._coef;
			}
		}

		if (iter % 2 == 0) {
			MPI_Waitall(2, send_req_1, MPI_STATUSES_IGNORE);
			MPI_Waitall(2, recv_req_1, MPI_STATUSES_IGNORE);
		} else {
			MPI_Waitall(2, send_req_2, MPI_STATUSES_IGNORE);
			MPI_Waitall(2, recv_req_2, MPI_STATUSES_IGNORE);
		}

		// Go through upper rows REDS
		if (id != 0) {
			for (std::size_t j = 2; j < n2 - 1; j += 2) {
				local_solution[j] =
					(eq._h_2 * eq._f(eq.x1[local_offset], eq.x2[j]) +
					 local_solution_prev[j - 1] + local_solution_prev[j + 1] +
					 row_above[j] + local_solution_prev[n2 + j]) /
					eq._coef;
			}
		}


		// Go through lower rows REDS
		if (id != num_procs - 1) {
			for (std::size_t j = 1 + reg_local_size % 2; j < n2 - 1; j += 2) {
				local_solution[(reg_local_size - 1) * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[(reg_local_size - 1) + local_offset],
									 eq.x2[j]) +
					 local_solution_prev[(reg_local_size - 1) * n2 + j - 1] +
					 local_solution_prev[(reg_local_size - 1) * n2 + j + 1] +
					 local_solution_prev[(reg_local_size - 2) * n2 + j] +
					 row_below[j]) /
					eq._coef;
			}
		}

		if (iter % 2 != 0) {
			MPI_Startall(2, send_req_1);
			MPI_Startall(2, recv_req_1);
		} else {
			MPI_Startall(2, send_req_2);
			MPI_Startall(2, recv_req_2);
		}

		// Go through inner BLACKS
		for (std::size_t i = 1; i < reg_local_size - 1; ++i) {
			for (std::size_t j = 1 + i % 2; j < n2 - 1; j += 2) {
				local_solution[i * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[i + local_offset], eq.x2[j]) +
					 local_solution[i * n2 + j - 1] +
					 local_solution[i * n2 + j + 1] +
					 local_solution[(i - 1) * n2 + j] +
					 local_solution[(i + 1) * n2 + j]) /
					eq._coef;
			}
		}

		if (iter % 2 != 0) {
			MPI_Waitall(2, send_req_1, MPI_STATUSES_IGNORE);
			MPI_Waitall(2, recv_req_1, MPI_STATUSES_IGNORE);
		} else {
			MPI_Waitall(2, send_req_2, MPI_STATUSES_IGNORE);
			MPI_Waitall(2, recv_req_2, MPI_STATUSES_IGNORE);
		}

		// Go through upper rows BLACKS
		if (id != 0) {
			for (std::size_t j = 1; j < n2 - 1; j += 2) {
				local_solution[j] =
					(eq._h_2 * eq._f(eq.x1[local_offset], eq.x2[j]) +
					 local_solution[j - 1] + local_solution[j + 1] +
					 row_above[j] + local_solution[n2 + j]) /
					eq._coef;
			}
		}


		// Go through lower rows BLACKS
		if (id != num_procs - 1) {
			for (std::size_t j = 1 + (reg_local_size - 1) % 2; j < n2 - 1;
				 j += 2) {
				local_solution[(reg_local_size - 1) * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[(reg_local_size - 1) + local_offset],
									 eq.x2[j]) +
					 local_solution[(reg_local_size - 1) * n2 + j - 1] +
					 local_solution[(reg_local_size - 1) * n2 + j + 1] +
					 local_solution[(reg_local_size - 2) * n2 + j] +
					 row_below[j]) /
					eq._coef;
			}
		}

		local_err =
			norm_difference(local_solution_prev, local_solution, reg_local_size,
							n2, config_Helmholtz_eq::NORM_TYPE, eps);

		MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		++iter;
	} while (sqrt(err) > eps);

	eq.time += MPI_Wtime();

	for (std::size_t i = 0; i < num_procs; ++i) {
		reg_sizes[i] *= n2;
		offsets[i] *= n2;
	}

	MPI_Gatherv(local_solution.data(), reg_local_size * n2, MPI_DOUBLE,
				eq.res.data(), reg_sizes.data(), offsets.data(), MPI_DOUBLE, 0,
				MPI_COMM_WORLD);

	delete[] send_req_1;
	delete[] send_req_2;
	delete[] recv_req_1;
	delete[] recv_req_2;

	return;
}


template <typename T>
void Seidel_RB_MPI_ISend_IRecv_Waitany(int id, int num_procs,
									   Helmholtz_equation<T>& eq,
									   std::size_t& iter, const T& eps = 1e-6) {
	iter = 1;
	std::size_t n1 = eq.x2.size(), n2 = eq.x1.size();

	MPI_Request* send_req_1;
	MPI_Request* send_req_2;
	MPI_Request* recv_req_1;
	MPI_Request* recv_req_2;

	/**
	 * Rows for computations for processor
	 */
	std::vector<int> reg_sizes;
	std::vector<int> offsets;
	int reg_local_size, local_offset;

	divide_4_regions(eq, id, num_procs, reg_local_size, local_offset, reg_sizes,
					 offsets);

	send_req_1 = new MPI_Request[2];
	send_req_2 = new MPI_Request[2];
	recv_req_1 = new MPI_Request[2];
	recv_req_2 = new MPI_Request[2];

	std::vector<T> local_solution(reg_local_size * n2, 0.);
	std::vector<T> local_solution_prev(reg_local_size * n2, 0.);
	std::vector<T> row_above(n2, 0.), row_below(n2, 0.);

	T err, local_err;

	int source = (id != 0) ? id - 1 : num_procs - 1;
	int dest = (id != num_procs - 1) ? id + 1 : 0;

	int send_count = (id != 0) ? n2 : 0;
	int recv_count = (id != num_procs - 1) ? n2 : 0;

	MPI_Send_init(local_solution_prev.data(), send_count, MPI_DOUBLE, source,
				  100, MPI_COMM_WORLD, send_req_1);
	MPI_Recv_init(row_below.data(), recv_count, MPI_DOUBLE, dest, MPI_ANY_TAG,
				  MPI_COMM_WORLD, recv_req_1);

	MPI_Send_init(local_solution_prev.data() + (reg_local_size - 1) * n2,
				  recv_count, MPI_DOUBLE, dest, 100, MPI_COMM_WORLD,
				  send_req_1 + 1);
	MPI_Recv_init(row_above.data(), send_count, MPI_DOUBLE, source, MPI_ANY_TAG,
				  MPI_COMM_WORLD, recv_req_1 + 1);

	MPI_Send_init(local_solution.data(), send_count, MPI_DOUBLE, source, 100,
				  MPI_COMM_WORLD, send_req_2);
	MPI_Recv_init(row_below.data(), recv_count, MPI_DOUBLE, dest, MPI_ANY_TAG,
				  MPI_COMM_WORLD, recv_req_2);

	MPI_Send_init(local_solution.data() + (reg_local_size - 1) * n2, recv_count,
				  MPI_DOUBLE, dest, 100, MPI_COMM_WORLD, send_req_2 + 1);
	MPI_Recv_init(row_above.data(), send_count, MPI_DOUBLE, source, MPI_ANY_TAG,
				  MPI_COMM_WORLD, recv_req_2 + 1);

	/**
	 * Main body of Seidel_RB method
	 */
	do {
		local_solution.swap(local_solution_prev);

		if (iter % 2 == 0) {
			MPI_Startall(2, send_req_1);
			MPI_Startall(2, recv_req_1);
		} else {
			MPI_Startall(2, send_req_2);
			MPI_Startall(2, recv_req_2);
		}


		// Go through inner REDS
		for (std::size_t i = 1; i < reg_local_size - 1; ++i) {
			for (std::size_t j = 1 + (i + 1) % 2; j < n2 - 1; j += 2) {
				local_solution[i * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[i + local_offset], eq.x2[j]) +
					 local_solution_prev[i * n2 + j - 1] +
					 local_solution_prev[i * n2 + j + 1] +
					 local_solution_prev[(i - 1) * n2 + j] +
					 local_solution_prev[(i + 1) * n2 + j]) /
					eq._coef;
			}
		}

		for (auto k : {0, 1}) {
			int ind1, ind2;
			if (iter % 2 == 0) {
				MPI_Waitany(2, send_req_1, &ind1, MPI_STATUSES_IGNORE);
				MPI_Waitany(2, recv_req_1, &ind2, MPI_STATUSES_IGNORE);
			} else {
				MPI_Waitany(2, send_req_2, &ind1, MPI_STATUSES_IGNORE);
				MPI_Waitany(2, recv_req_2, &ind2, MPI_STATUSES_IGNORE);
			}

			if (ind1 == 1) {
				// Go through upper rows REDS
				if (id != 0) {
					for (std::size_t j = 2; j < n2 - 1; j += 2) {
						local_solution[j] =
							(eq._h_2 * eq._f(eq.x1[local_offset], eq.x2[j]) +
							 local_solution_prev[j - 1] +
							 local_solution_prev[j + 1] + row_above[j] +
							 local_solution_prev[n2 + j]) /
							eq._coef;
					}
				}
			}
			if (ind2 == 0) {
				// Go through lower rows REDS
				if (id != num_procs - 1) {
					for (std::size_t j = 1 + (reg_local_size - 1) % 2;
						 j < n2 - 1; j += 2) {
						for (std::size_t j = 1 + reg_local_size % 2; j < n2 - 1;
							 j += 2) {
							local_solution[(reg_local_size - 1) * n2 + j] =
								(eq._h_2 * eq._f(eq.x1[(reg_local_size - 1) +
													   local_offset],
												 eq.x2[j]) +
								 local_solution_prev[(reg_local_size - 1) * n2 +
													 j - 1] +
								 local_solution_prev[(reg_local_size - 1) * n2 +
													 j + 1] +
								 local_solution_prev[(reg_local_size - 2) * n2 +
													 j] +
								 row_below[j]) /
								eq._coef;
						}
					}
				}
			}
		}

		if (iter % 2 != 0) {
			MPI_Startall(2, send_req_1);
			MPI_Startall(2, recv_req_1);
		} else {
			MPI_Startall(2, send_req_2);
			MPI_Startall(2, recv_req_2);
		}

		// Go through inner BLACKS
		for (std::size_t i = 1; i < reg_local_size - 1; ++i) {
			for (std::size_t j = 1 + i % 2; j < n2 - 1; j += 2) {
				local_solution[i * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[i + local_offset], eq.x2[j]) +
					 local_solution[i * n2 + j - 1] +
					 local_solution[i * n2 + j + 1] +
					 local_solution[(i - 1) * n2 + j] +
					 local_solution[(i + 1) * n2 + j]) /
					eq._coef;
			}
		}

		for (auto k : {0, 1}) {
			int ind1, ind2;
			if (iter % 2 != 0) {
				MPI_Waitany(2, send_req_1, &ind1, MPI_STATUSES_IGNORE);
				MPI_Waitany(2, recv_req_1, &ind2, MPI_STATUSES_IGNORE);
			} else {
				MPI_Waitany(2, send_req_2, &ind1, MPI_STATUSES_IGNORE);
				MPI_Waitany(2, recv_req_2, &ind2, MPI_STATUSES_IGNORE);
			}

			if (ind1 == 1) {
				// Go through upper rows BLACKS
				if (id != 0) {
					for (std::size_t j = 1; j < n2 - 1; j += 2) {
						local_solution[j] =
							(eq._h_2 * eq._f(eq.x1[local_offset], eq.x2[j]) +
							 local_solution[j - 1] + local_solution[j + 1] +
							 row_above[j] + local_solution[n2 + j]) /
							eq._coef;
					}
				}
			}
			if (ind2 == 0) {
				// Go through lower rows BLACKS
				if (id != num_procs - 1) {
					for (std::size_t j = 1 + (reg_local_size - 1) % 2;
						 j < n2 - 1; j += 2) {
						local_solution[(reg_local_size - 1) * n2 + j] =
							(eq._h_2 *
								 eq._f(
									 eq.x1[(reg_local_size - 1) + local_offset],
									 eq.x2[j]) +
							 local_solution[(reg_local_size - 1) * n2 + j - 1] +
							 local_solution[(reg_local_size - 1) * n2 + j + 1] +
							 local_solution[(reg_local_size - 2) * n2 + j] +
							 row_below[j]) /
							eq._coef;
					}
				}
			}
		}

		local_err =
			norm_difference(local_solution_prev, local_solution, reg_local_size,
							n2, config_Helmholtz_eq::NORM_TYPE, eps);

		MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		++iter;
	} while (sqrt(err) > eps);


	for (std::size_t i = 0; i < num_procs; ++i) {
		reg_sizes[i] *= n2;
		offsets[i] *= n2;
	}

	MPI_Gatherv(local_solution.data(), reg_local_size * n2, MPI_DOUBLE,
				eq.res.data(), reg_sizes.data(), offsets.data(), MPI_DOUBLE, 0,
				MPI_COMM_WORLD);

	delete[] send_req_1;
	delete[] send_req_2;
	delete[] recv_req_1;
	delete[] recv_req_2;

	return;
}
