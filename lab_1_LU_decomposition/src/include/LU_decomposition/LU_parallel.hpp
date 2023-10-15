#pragma once

#include <omp.h>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <ostream>
#include <vector>

#include "../Matrix_functions.hpp"

/**
 * Help function to avoid "-0." in array
 */
template <typename T>
void parallel_check_vector_zero(std::vector<T>& vec) {
	const T _eps = 1e-14;
#pragma omp parallel for
	for (std::size_t i = 0; i < vec.size(); ++i) {
		if (fabs(vec[i]) < _eps) {
			vec[i] = 0.;
			vec[i] *= (1 / vec[i] > 0) ? (1) : (-1);
		}
	}
}

/**
 * Parallel LU decomposition for n x m matrix
 * via OpenMP
 *
 * Variables:
 *      matrix --- matrix size [n x m]
 *      n --- matrix rows
 *      m --- matrix columns
 */
template <typename T>
void paral_lu_decomp(std::vector<T>& matrix, std::size_t n, std::size_t m) {
	for (std::size_t i = 0; i < std::min(n - 1, m); ++i) {
#pragma omp parallel for shared(matrix, n, m)
		for (std::size_t j = i + 1; j < n; ++j) {
			matrix[j * m + i] = matrix[j * m + i] / matrix[i * m + i];
		}

		if (i < n) {
#pragma omp parallel for shared(matrix, n, m)
			for (std::size_t j = i + 1; j < n; ++j) {
				for (std::size_t k = i + 1; k < m; ++k) {
					matrix[j * m + k] = matrix[j * m + k] -
										matrix[j * m + i] * matrix[i * m + k];
				}
			}
		}
	}
	parallel_check_vector_zero(matrix);	 // this function can be excluded
	return;
}


/**
 * Parallel block LU decomposition for n x n matrix
 * via OpenMP
 *
 * Variables:
 *      matrix --- matrix size [n x m]
 *      n --- matrix rows
 *      m --- matrix columns
 *      b --- block param (by default: b = 64)
 */
template <typename T>
void parallel_block_lu_decomp(std::vector<T>& matrix, const std::size_t n,
							  const std::size_t m, const std::size_t b = 64) {
	std::size_t step = b;

	/*
			   _____ = b
			 |[b b b * * * * * * *] |
		     b = |[b b b * * * * * * *] |
			 |[b b b * * * * * * *] |
			  [* * * * * * * * * *] |
		matrix  = [* * * * * * * * * *] | = n
			  [* * * * * * * * * *] |
			  [* * * * * * * * * *] |
			  [* * * * * * * * * *] |
			  [* * * * * * * * * *] |
			  --------------------- = m
		n --- matrix rows
		m --- matrix columns
		b --- size of block inside matrix

	*/
	/**
		Functon to get LU decomposion of matrix A
		which consists of b_block (size [n x n]) and
		under_b_block (size [m x n])
		Though matrix A is size of (m + n) x n
	*/
	auto get_LU_decomposition_for_b_block_and_under_b_block =
		[](std::vector<T>& b_block, std::vector<T>& under_b_block,
		   const std::size_t& m, const std::size_t& n) {
			for (std::size_t i = 0; i < std::min(m + n - 1, n); ++i) {
#pragma omp parallel for shared(b_block, under_b_block, m, n)
				for (std::size_t j = i + 1; j < m + n; ++j) {
					if (j < n) {
						b_block[j * n + i] /= b_block[i * n + i];
					} else {
						under_b_block[(j - n) * n + i] /= b_block[i * n + i];
					}
				}
				if (i < n) {
#pragma omp parallel for shared(b_block, under_b_block, m, n)
					for (std::size_t j = i + 1; j < m + n; ++j) {
						for (std::size_t k = i + 1; k < n; ++k) {
							if (j < n) {
								b_block[j * n + k] -=
									b_block[j * n + i] * b_block[i * n + k];
							} else {
								under_b_block[(j - n) * n + k] -=
									under_b_block[(j - n) * n + i] *
									b_block[i * n + k];
							}
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


	std::vector<T> b_block;
	std::vector<T> under_b_block;
	std::vector<T> right_b_block;
	/**
		Reserve memory for b_block + under_b_block
		and for right_b_block
	*/
	b_block.reserve(sub_step * sub_step);
	under_b_block.reserve((n - sub_step) * sub_step);
	right_b_block.reserve(sub_step * (m - sub_step));

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

		b_block.resize(step * step);
		under_b_block.resize((n - i - step) * step);
		right_b_block.resize(step * (m - i - step));


		/**
			Get LU_decomposition of this (b & u)-block of matrix
				[b b b * * * * * * *]
				[b b b * * * * * * *]
				[b b b * * * * * * *]
				[u u u * * * * * * *]
				[u u u * * * * * * *]
				[u u u * * * * * * *]
				[u u u * * * * * * *]
				[u u u * * * * * * *]
				[u u u * * * * * * *]
				[u u u * * * * * * *]
				[u u u * * * * * * *]
		*/
		// Copy matrix to b_block, under_b_block and right_b_block
#pragma omp parallel for shared(matrix, b_block, under_b_block, right_b_block, \
									step, n, m)
		for (std::size_t j = 0; j < (m - i); ++j) {
			for (std::size_t k = 0; k < step; ++k) {
				if (j < step && j < n - i) {
					b_block[j * step + k] = matrix[(i + j) * m + k + i];
				} else {
					if (j < n - i) {
						under_b_block[(j - step) * step + k] =
							matrix[(i + j) * m + k + i];
					}
					if (j < m - i)
						right_b_block[k * (m - i - step) + j - step] =
							matrix[(i + k) * m + j + i];
				}
			}
		}

		get_LU_decomposition_for_b_block_and_under_b_block(
			b_block, under_b_block, n - i - step, step);


		/**
			We need to get further LU decomposion
			(get_LU_decomposition_for_b_block_and_under_b_block) while our last
			block wouldn't be same size as L22
		*/
		if (i == n - step) {
#pragma omp parallel for shared(matrix, b_block, under_b_block, step, n, m)
			for (std::size_t j = 0; j < step; ++j) {
				for (std::size_t k = 0; k < step; ++k) {
					matrix[(i + j) * m + k + i] = b_block[j * step + k];
				}
			}

			if (n < m) {
				// Get LU_decomposition of right_b_block (like in
				// further part of the code)
#pragma omp parallel for shared(b_block, right_b_block, step, n, m)
				for (std::size_t j = 1; j < step; ++j) {
					for (std::size_t l = 0; l < step - j; ++l) {
						for (std::size_t k = 0; k < m - i - step; ++k) {
							right_b_block[(j + l) * (m - i - step) + k] -=
								b_block[(j + l) * step + j - 1] *
								right_b_block[(j - 1) * (m - i - step) + k];
						}
					}
				}
				// Copy right_b_block to matrix
#pragma omp parallel for shared(matrix, right_b_block, step, n, m)
				for (std::size_t j = 0; j < step; ++j) {
					for (std::size_t k = 0; k < m - i - step; ++k) {
						matrix[(i + j) * m + k + i + step] =
							right_b_block[j * (m - i - step) + k];
					}
				}
			}

			break;
		}


#pragma omp parallel for shared(matrix, b_block, under_b_block, step, n, m)
		for (std::size_t j = 0; j < (n - i); ++j) {
			for (std::size_t k = 0; k < step; ++k) {
				if (j < step) {
					matrix[(i + j) * m + k + i] = b_block[j * step + k];
				} else {
					matrix[(i + j) * m + k + i] =
						under_b_block[(j - step) * step + k];
				}
			}
		}

		/**
			Find U23 = L22^(-1) * A23
			via straight part of Gauss algorithm
			(
				As L22 is lower triangular matrix, we can get U23
				by removing elements under L22 diagonal and jointly
				changing A23
			)

			Get LU_decomposition of block of A23
				[b b b r r r r r r r]
				[b b b r r r r r r r]
				[b b b r r r r r r r]
				[u u u * * * * * * *]
				[u u u * * * * * * *]
				[u u u * * * * * * *]
				[u u u * * * * * * *]
				[u u u * * * * * * *]
				[u u u * * * * * * *]
				[u u u * * * * * * *]
				[u u u * * * * * * *]
		*/
#pragma omp parallel for shared(b_block, right_b_block, step, n, m)
		for (std::size_t j = 1; j < step; ++j) {
			for (std::size_t l = 0; l < step - j; ++l) {
				for (std::size_t k = 0; k < m - i - step; ++k) {
					right_b_block[(j + l) * (m - i - step) + k] -=
						b_block[(j + l) * step + j - 1] *
						right_b_block[(j - 1) * (m - i - step) + k];
				}
			}
		}
		// Copy right_b_block to matrix
#pragma omp parallel for shared(matrix, right_b_block, step, n, m)
		for (std::size_t j = 0; j < step; ++j) {
			for (std::size_t k = 0; k < m - i - step; ++k) {
				matrix[(i + j) * m + k + i + step] =
					right_b_block[j * (m - i - step) + k];
			}
		}

		/**
			Find A33 = A33 - L23 * U23;
			via fast IKJ algorithm

			Update untouched part (* -> o):
				[b b b r r r r r r r]
				[b b b r r r r r r r]
				[b b b r r r r r r r]
				[u u u o o o o o o o]
				[u u u o o o o o o o]
				[u u u o o o o o o o]
				[u u u o o o o o o o]
				[u u u o o o o o o o]
				[u u u o o o o o o o]
				[u u u o o o o o o o]
		*/
#pragma omp parallel for shared(matrix, under_b_block, right_b_block, step, n, \
									m)
		for (std::size_t l = 0; l < n - i - step; ++l) {
			for (std::size_t k = 0; k < step; ++k) {
				for (std::size_t j = 0; j < m - i - step; ++j) {
					matrix[(l + i + step) * m + (j + i + step)] -=
						under_b_block[l * step + k] *
						right_b_block[k * (m - i - step) + j];
				}
			}
		}

		/**
			Recursively repeat algorithm with o-matrix
			(# --- decomposed part of matrix):

				[# # # # # # # # # #]
				[# # # # # # # # # #]
				[# # # # # # # # # #]
				[# # # o o o o o o o]
				[# # # o o o o o o o]
				[# # # o o o o o o o]
				[# # # o o o o o o o]
				[# # # o o o o o o o]
				[# # # o o o o o o o]
				[# # # o o o o o o o]
		*/
	}
	parallel_check_vector_zero(matrix);	 // this function can be excluded
	return;
}
