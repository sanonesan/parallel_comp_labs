#include <omp.h>

#include <cstddef>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>

#include "./include/LU_decomposition/LU_sequetial.hh"


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
	typedef double T;
	
	std::size_t n = 3;
	std::vector<T> A = {2., -5., 1., -1., 3., -1., 3., -4., 2.};

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

	print_matrix(A, 3);

	lu_seq_decomp(A, 3);

    for (std::size_t i = 1; i < n; ++i){
        for (std::size_t j = 0; j < i; ++j) {
            L[i * n + j] = A[i * n + j];
        }
    }
    for (std::size_t i = 0; i < n; ++i){
        for (std::size_t j = i; j < n; ++j) {
            U[i * n + j] = A[i * n + j];
        }
    }

	print_matrix(L, 3);
	print_matrix(U, 3);


    std::vector<T> A_test;

	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < n; ++j) {
            A_test.push_back(0.);
        }
	}   

    for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < n; ++k) {
                A_test[i * n + j] += L[i * n + k] * U[k * n + j];
            }
        }
	}   
	

    print_matrix(A_test, n);
    
	return 0;
}
