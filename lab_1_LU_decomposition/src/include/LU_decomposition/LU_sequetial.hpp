#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "../Matrix_functions.hpp"

/**
 * Sequential LU decomposition for n x n matrix
 *
 * Variables:
 *      matrix --- matrix size [n x n]
 *      n --- matrix size
 */
template <typename T>
void lu_seq_decomp(std::vector<T>& matrix, std::size_t n) {
	for (std::size_t i = 0; i < n - 1; ++i) {
		for (std::size_t j = i + 1; j < n; ++j) {
			matrix[j * n + i] = matrix[j * n + i] / matrix[i * n + i];
		}
		for (std::size_t j = i + 1; j < n; ++j) {
			for (std::size_t k = i + 1; k < n; ++k) {
				matrix[j * n + k] -= matrix[j * n + i] * matrix[i * n + k];
				// matrix[j * n + k] - matrix[j * n + i] * matrix[i * n + k];
			}
		}
	}
	check_vector_zero(matrix);
	return;
}


/**
 * Sequential block LU decomposition for n x n matrix
 *
 * Variables:
 *      matrix --- matrix size [n x n]
 *      n --- matrix size
 *      b --- block param (by default: b = 32)
 */
template <typename T>
void block_lu_decomp(std::vector<T>& matrix, std::size_t n,
					 std::size_t b = 32) {
	std::size_t step = b;


	/**
		Functon to get LU decomposion of matrix A (size m x v) inside
		of main matrix.

		ii - index in main matrix wich points on the beggining of A
			 (matrix[ii][ii] = A[0, 0])
	*/
	auto algorithm_2_9 = [n](std::vector<T>& A, std::size_t ii,
							 std::size_t step) {
		std::size_t m = n - ii;
		std::size_t v = step;

		for (std::size_t i = ii; i < ii + std::min(m - 1, v); ++i) {
			for (std::size_t j = i + 1; j < m + ii; ++j) {
				A[j * n + i] /= A[i * n + i];
			}
			if (i - ii < v) {
				for (std::size_t j = i + 1; j < m + ii; ++j) {
					for (std::size_t k = i + 1; k < v + ii; ++k) {
						A[j * n + k] -= A[j * n + i] * A[i * n + k];
					}
				}
			}
		}
		return;
	};


	/**
		Adding sub step for first iteration to avoid
		run out of index problem
		(
		For example:
			If size of b_block = 4 and matrix is size of 10x10
			10 % 4 = 2.
			Thus we need to make first step size of 2 or 6
		)
	*/
	bool flag = true;
	std::size_t sub_step = step;

	if (n % step != 0) {
		flag = false;
		sub_step = n % step;
		if (step - sub_step < 2) {
			sub_step += step;
		}
	}

	/**
		Main body of block LU decomposion algorithm
	*/
	for (std::size_t i = 0; i < n - 1; i += step) {
		/**
			Do first iteration with sub_step if needed
		*/
		if (flag == false && i == 0) {
			step = sub_step;
		} else if (flag == false && i != 0) {
			flag = true;
			step = b;
		} else if (flag == true && i == 0) {
			step = b;
		}

		algorithm_2_9(matrix, i, step);
		/**
			We need to get further LU decomposion (algorithm_2_9)
			while our last block wouldn't be same size as L22
		*/
		if (i == n - step) {
			break;
		}

		/**
			Find U23 = L22^(-1) * A23
			via straight part of Gauss algorithm
			(
				As L22 is lower triangular matrix, we can get U23
				by removing elements under L22 diagonal and jointly
				changing A23
			)
		*/
		for (std::size_t j = i + 1; j < i + step; ++j) {
			for (std::size_t l = 0; l < step - j + i; ++l) {
				for (std::size_t k = i + step; k < n; ++k) {
					matrix[(j + l) * n + k] -=
						matrix[(j + l) * n + j - 1] * matrix[(j - 1) * n + k];
				}
			}
		}

		/**
			Find A33 = A33 - L23 * U23;
			via fast KIJ algorithm
		*/
		for (std::size_t k = 0; k < step; ++k) {
			for (std::size_t l = i + step; l < n; ++l) {
				for (std::size_t j = i + step; j < n; ++j) {
					matrix[l * n + j] -=
						matrix[l * n + (k + i)] * matrix[(i + k) * n + j];
				}
			}
		}
	}

	check_vector_zero(matrix);

	return;
}
