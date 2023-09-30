#include <time.h>

#include <cstddef>
#include <iostream>
#include <iterator>
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

	std::size_t n = 1024;
	std::vector<T> B;  // = {2., -5., 1., -1., 3., -1., 3., -4., 2.};
	B.reserve(n * n);
	fill_martrix_with_random_numbers(B, n, 50, 1);	//, 1, 2);

	std::vector<T> A(B);  //= {1, 0, 0, -2, 1, 0, -3, -2, 1};	//(B);

	//
	// std::size_t i = 0, step = 3;
	//
	//
	// print_matrix(B, n);
	// print_matrix(A, n);


	// for (std::size_t j = i + 1; j < i + step; ++j) {
	// 	// for (std::size_t k = i + step; k < n; ++k) {
	// 	// for (std::size_t l = step - j + i; l > 0; --l) {
	// 	for (std::size_t l = 0; l < step - j; ++l) {
	// 		for (std::size_t k = i + step; k < n; ++k) {
	// 			printf("B[%d, %d] = %f \t", j + l, k, B[(j + l) * n + k]);
	// 			printf("B[%d, %d] = %f \t ", j + l, j - 1,
	// 				   B[(j + l) * n + j - 1]);
	// 			printf("B[%d, %d] = %f \n", j - 1, k, B[(j - 1) * n + k]);
	// 			B[(j + l) * n + k] -=
	// 				B[(j + l) * n + j - 1] * B[(j - 1) * n + k];
	// 			// printf("B[%d, %d] = %f \n", j, k, B[j * n + k]);
	// 		}
	// 		print_matrix(B, n);
	// 	}
	// }
	// for (std::size_t j = i + 1; j < i + step; ++j) {
	// 	for (std::size_t l = step - j + i; l > 0; --l) {
	// 		for (std::size_t k = i + step; k < n; ++k) {
	// 			printf("B[%d, %d] = %f \t \n", j - 1, k, B[(j - 1) * n + k]);
	// 		}
	// 		printf("\n");
	// 	}
	// 	printf("\n");
	// }

	// for (std::size_t j = i + 1; j < i + step; ++j) {
	// 	for (std::size_t k = i + step; k < n; ++k) {
	// 		for (std::size_t l = 0; l < j - i; ++l) {
	// 		}
	// 		B[j * n + k] -= B[(j - 1) * n + k] * B[j * n + j];
	// 	}
	// }

	// print_matrix(B, n);


	double t1, t2;
	// print_matrix(A, n);
	t1 = omp_get_wtime();
	//
	lu_paral_decomp(A, n);
	// print_matrix(A, n);


	t2 = omp_get_wtime();
	//
	std::cout << "Paral Time taken: " << t2 - t1 << "\n";
	//

	A = B;
	//
	t1 = omp_get_wtime();
	//
	lu_seq_decomp(A, n);
	// print_matrix(A, n);
	//
	t2 = omp_get_wtime();

	// print_matrix(K, n);

	std::cout << "Seq Time taken: " << t2 - t1 << "\n";
	//
	//
	A = B;
	//
	t1 = omp_get_wtime();
	block_lu_decomp(A, n, 64);
	// print_matrix(A, n, n);
	t2 = omp_get_wtime();
	//
	std::cout << "Seq Block Time taken: " << t2 - t1 << "\n";
	//

	// std::cout << "Seq Time taken: " << t2 - t1 << "\n";
	//
	// t1 = omp_get_wtime();
	// matrix_mult(A, B, n);
	// t2 = omp_get_wtime();
	// print_matrix(C, n);
	// print_matrix(A, n, n);	// t2 = omp_get_wtime();
	//
	// std::vector<T> L(A), U(A);
	// get_LU_matrices(A, L, U, n);
	// std::cout << "AAAA\n";
	// auto K = matrix_mult(L, U, n);
	// //

	// std::cout << "Seq Time taken: " << t2 - t1 << "\n";

	return 0;
}
