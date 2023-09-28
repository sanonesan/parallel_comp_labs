#pragma once

#include <iomanip>
#include <iostream>
#include <vector>


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


template <typename T>
std::vector<T> matrix_mult(std::vector<T>& A, std::vector<T>& B,
						   std::size_t n) {
	std::vector<T> Result;
	Result.reserve(n * n);
	Result.resize(n * n);
	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < n; ++j) {
			for (std::size_t k = 0; k < n; ++k) {
				Result[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
	return Result;
}


template <typename T>
void get_LU_matrices(std::vector<T>& matrix, std::vector<T>& L,
					 std::vector<T>& U, std::size_t n) {
	L.resize(n * n);
	U.resize(n * n);

	for (std::size_t i = 0; i < n * n; ++i) {
		L.push_back(0.);
		U.push_back(0.);
	}
	for (std::size_t i = 0; i < n; ++i) {
		L[i * n + i] = 1.;
	}

	for (std::size_t i = 1; i < n; ++i) {
		for (std::size_t j = 0; j < i; ++j) {
			L[i * n + j] = matrix[i * n + j];
		}
	}
	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = i; j < n; ++j) {
			U[i * n + j] = matrix[i * n + j];
		}
	}

	return;
}
