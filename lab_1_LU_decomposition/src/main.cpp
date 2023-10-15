#include <omp.h>

#include <cstddef>
#include <fstream>
#include <iterator>
#include <vector>

#include "./config.hpp"
#include "./include/LU_decomposition/LU_parallel.hpp"
#include "./include/LU_decomposition/LU_sequetial.hpp"
#include "./include/Matrix_functions.hpp"
#include "./include/Matrix_generator.hpp"


int main(int argc, char* argv[]) {
	typedef double T;

	std::srand((unsigned int)time(NULL));

	std::size_t n = 2048;
	std::vector<T> B;
	B.reserve(n * n);
	fill_martrix_with_random_numbers(B, n, 50, 1);
	std::vector<T> A(B);

	// std::ofstream output_file("../output/output_2048.txt");
	std::ofstream output_file(config_lab_1::PATH_output_folder +
							  "/output_2048.txt");

	double t1, t2;

	/**
	 *
	 * Sequential algorithms section
	 *
	 */

	output_file << "Sequential part: \n";
	A = B;
	t1 = omp_get_wtime();
	sequential_lu_decomp(A, n, n);
	t2 = omp_get_wtime();
	output_file << std::setw(30) << "Sequential LU_dec time: " << std::setw(9)
				<< t2 - t1 << "\n";

	A = B;
	t1 = omp_get_wtime();
	sequential_block_lu_decomp(A, n, n, 64);
	t2 = omp_get_wtime();
	output_file << std::setw(30)
				<< "Sequetial Block LU_dec time: " << std::setw(9) << t2 - t1
				<< "\n";


	/**
	 *
	 * Parallel algorithms section
	 *
	 */

	output_file << "\nParallel part: \n";
	std::vector<std::size_t> threads_vec = {1, 2, 4, 6};  //, 8, 12, 16, 18};

	for (auto num_threads = threads_vec.begin();
		 num_threads != threads_vec.end(); ++num_threads) {
		std::cout << "\nStart:\tnum_threads = " << *num_threads << "\n";
		omp_set_num_threads(*num_threads);

		A = B;
		output_file << "num_threads = " << *num_threads << "\n";
		t1 = omp_get_wtime();
		paral_lu_decomp(A, n, n);
		t2 = omp_get_wtime();
		output_file << std::setw(30) << "Parallel LU_dec time: " << std::setw(9)
					<< t2 - t1 << "\n";
		A = B;
		t1 = omp_get_wtime();
		parallel_block_lu_decomp(A, n, n, 64);
		t2 = omp_get_wtime();
		output_file << std::setw(30)
					<< "Parallel Block LU_dec time: " << std::setw(9) << t2 - t1
					<< "\n";
		std::cout << "\nFinish:\tnum_threads = " << *num_threads << "\n";
	}

	output_file.close();
	/**
	 * Check LU decomposition
	 */

	// C = matrix_mult(A, B, n);
	// print_matrix(C, n);
	// print_matrix(A, n, n);


	/**
	 * Get L & U matrices
	 */
	// std::vector<T> L(A), U(A);
	// get_LU_matrices(A, L, U, n);
	// std::cout << "AAAA\n";
	// auto K = matrix_mult(L, U, n);

	return 0;
}
