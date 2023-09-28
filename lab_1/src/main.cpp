#include <time.h>

#include <cstddef>
#include <iostream>
#include <vector>

#include "./include/LU_decomposition/LU_parallel.hh"
#include "./include/LU_decomposition/LU_sequetial.hh"
#include "./include/Matrix_functions.hh"
#include "./include/Matrix_generator.hh"
#include "omp.h"

int main(int argc, char* argv[]) {
	clock_t t_start, t_end;

	typedef double T;

	std::srand((unsigned int)time(NULL));


	std::size_t n = 3;
	std::vector<T> B;  // = {2., -5., 1., -1., 3., -1., 3., -4., 2.};
	B.reserve(n * n);
	fill_martrix_with_random_numbers(B, n);	 //, 1, 2);

	std::vector<T> A(B);

	double t1, t2;

	t1 = omp_get_wtime();

	lu_paral_decomp(A, n);

	t2 = omp_get_wtime();

	std::cout << "Paral Time taken: " << t2 - t1 << "\n";


	A = B;


	t1 = omp_get_wtime();

	lu_seq_decomp(A, n);

	t2 = omp_get_wtime();

	std::cout << "Seq Time taken: " << t2 - t1 << "\n";

	return 0;
}
