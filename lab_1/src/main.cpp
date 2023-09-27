#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>
#include <random>

#include "omp.h"
#include <time.h>

#include "./include/LU_decomposition/LU_sequetial.hh"
#include "./include/LU_decomposition/LU_parallel.hh"

/**
*   Print 1d array in Matrix form
*/
template <typename T>
void print_matrix(std::vector<T>& matrix, std::size_t n) {
	std::cout << "[\n";
	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < n; ++j) {
			std::cout << std::setw(15) << matrix[i * n + j] << ", ";
		}
		std::cout << "\n";
	}
	std::cout << "]\n";

	return;
}

int main(int argc, char* argv[]) {

    clock_t t_start, t_end;

	typedef double T;

    std::srand((unsigned int)time(NULL));

	
	std::size_t n = 1000;
	std::vector<T> B; // = {2., -5., 1., -1., 3., -1., 3., -4., 2.};
    
    for (std::size_t i = 0; i < n * n; ++i)
        if ((rand() % (10 - 1 + 1) + 1) > 5)
            B.push_back( rand() % (50 - 1 + 1) + 1);
        else 
            B.push_back( - (rand() % (50 - 1 + 1) + 1));
    
    std::vector<T> A(B);

	std::vector<T> L, U;
	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < n; ++j) {
			L.push_back(0.);
			U.push_back(0.);
        }
	}

	for (std::size_t i = 0; i < n; ++i) {
		L[i * n + i] = 1.;
	}

	// print_matrix(A, n); 
    
    double t1, t2;

    t1 = omp_get_wtime();

    lu_paral_decomp(A, n);

    t2 = omp_get_wtime();

    std::cout << "Paral Time taken: " <<  t2 - t1 << "\n";


    A = B;


    t1 = omp_get_wtime();

    lu_seq_decomp(A, n);

    t2 = omp_get_wtime();

    std::cout << "Seq Time taken: " << t2 - t1 << "\n";


 //
 //
 // 
 //
	// print_matrix(A, n); 
 //    for (std::size_t i = 1; i < n; ++i){
 //        for (std::size_t j = 0; j < i; ++j) {
 //            L[i * n + j] = A[i * n + j];
 //        }
 //    }
 //    for (std::size_t i = 0; i < n; ++i){
 //        for (std::size_t j = i; j < n; ++j) {
 //            U[i * n + j] = A[i * n + j];
 //        }
 //    }
 //
	// print_matrix(L, n);
	// print_matrix(U, n);
 //
 //
 //    std::vector<T> A_test;
 //
	// for (std::size_t i = 0; i < n; ++i) {
	// 	for (std::size_t j = 0; j < n; ++j) {
 //            A_test.push_back(0.);
 //        }
	// }   
 //
 //    for (std::size_t i = 0; i < n; ++i) {
	// 	for (std::size_t j = 0; j < n; ++j) {
 //            for (std::size_t k = 0; k < n; ++k) {
 //                A_test[i * n + j] += L[i * n + k] * U[k * n + j];
 //            }
 //        }
	// }   
 //
 //
 //    print_matrix(A_test, n);
 //    
	return 0;
}
