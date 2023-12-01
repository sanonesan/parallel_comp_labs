#pragma once

#include <mpi.h>

#include <vector>

#include "../class_Helmholtz.hpp"

template <typename T>
void divide_4_regions(Helmholtz_equation<T>& eq, int id, int num_procs,
					  int& reg_local_size, int& local_offset,
					  std::vector<int>& reg_sizes, std::vector<int>& offsets) {
	/**
	 * eq --- Helmholtz_equation
	 * id --- processor include
	 * num_procs --- number of processor
	 * reg_local_size --- size of local region (after division)
	 * local_offset --- offset of region (number of row where processor
	 *                  takes its values)
	 * reg_sizes --- vector of divided regions' reg_sizes
	 * offsets --- vector of regions' offsets
	 */

	reg_sizes.resize(num_procs, 0);
	offsets.resize(num_procs);

	for (std::size_t i = 0; i < num_procs - 1; ++i) {
		reg_sizes[i] = eq.x1.size() / num_procs;
		offsets[i + 1] = offsets[i] + reg_sizes[i];
	}
	reg_sizes[num_procs - 1] =
		eq.x1.size() / num_procs + eq.x1.size() % num_procs;
	/**
	 *
	 */
	MPI_Scatter(reg_sizes.data(), 1, MPI_INT, &reg_local_size, 1, MPI_INT, 0,
				MPI_COMM_WORLD);

	MPI_Scatter(offsets.data(), 1, MPI_INT, &local_offset, 1, MPI_INT, 0,
				MPI_COMM_WORLD);
};
