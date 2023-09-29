#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iterator>
#include <utility>
#include <vector>

#include "../Matrix_functions.hh"

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
 * matrix --- матрица
 * n --- размерность матрицы
 * b --- блочный параметр (by default: b = 2)
 *
 */
template <typename T>
void block_lu_decomp(std::vector<T>& matrix, std::size_t n, std::size_t b = 2) {
	std::size_t step = b;

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


	bool flag = true;
	std::size_t sub_step = step;

	if (n % step != 0) {
		flag = false;
		sub_step += n % step;
	}

	std::vector<T> L22_block, identity_b_block, lower_b_block, upper_b_block;
	std::size_t reserve_min = 0;

	reserve_min = std::max(sub_step, n - sub_step);

	L22_block.reserve(sub_step * sub_step);
	identity_b_block.reserve(sub_step * sub_step);

	lower_b_block.reserve(reserve_min * reserve_min);
	upper_b_block.reserve(reserve_min * reserve_min);


	std::size_t i = 0;
	while (i < n - 1) {
		if (flag == false && i == 0) {
			step = sub_step;
			L22_block.resize(step * step);
			upper_b_block.resize(step * (n - step));
			identity_b_block.resize(step * step);

			for (std::size_t i = 0; i < step; ++i) {
				for (std::size_t j = 0; j < step; ++j) {
					if (i == j)
						identity_b_block[i * step + j] = 1.;
					else
						identity_b_block[i * step + j] = 0.;
				}
			}
		} else if (flag == false && i != 0) {
			flag = true;
			step = b;
			L22_block.resize(step * step);
			upper_b_block.resize(step * (n - step));
			identity_b_block.resize(step * step);

			for (std::size_t i = 0; i < step; ++i) {
				for (std::size_t j = 0; j < step; ++j) {
					if (i == j)
						identity_b_block[i * step + j] = 1.;
					else
						identity_b_block[i * step + j] = 0.;
				}
			}
		} else if (flag == true && i == 0) {
			step = b;
			L22_block.resize(step * step);
			upper_b_block.resize(step * (n - step));
			identity_b_block.resize(step * step);

			for (std::size_t i = 0; i < step; ++i) {
				for (std::size_t j = 0; j < step; ++j) {
					if (i == j)
						identity_b_block[i * step + j] = 1.;
					else
						identity_b_block[i * step + j] = 0.;
				}
			}
		}

		algorithm_2_9(matrix, i, step);
		if (i == n - step) {
			break;
		}


		L22_block.assign(identity_b_block.begin(), identity_b_block.end());
		// get L22_block
		for (std::size_t j = i + 1; j < i + step; ++j) {
			for (std::size_t k = i; k < j; ++k) {
				L22_block[(j - i) * step + (k - i)] = matrix[j * n + k];
			}
		}

		for (std::size_t j = i; j < i + step; ++j) {
			for (std::size_t k = i + step; k < n; ++k) {
				upper_b_block[(j - i) * (n - step - i) + (k - i - step)] =
					matrix[j * n + k];
			}
		}

		L22_block = inverse_Matrix(L22_block, step);

		for (std::size_t l = i; l < i + step; ++l) {
			for (std::size_t j = i + step; j < n; ++j) {
				matrix[l * n + j] = 0.;
				for (std::size_t k = 0; k < step; ++k) {
					matrix[l * n + j] +=
						L22_block[(l - i) * step + k] *
						upper_b_block[k * (n - step - i) + (j - i - step)];
				}
			}
		}

		for (std::size_t l = i + step; l < n; ++l) {
			for (std::size_t j = i + step; j < n; ++j) {
				for (std::size_t k = 0; k < step; ++k) {
					matrix[l * n + j] -=
						matrix[l * n + (k + i)] * matrix[(i + k) * n + j];
				}
			}
		}
		i += step;
	}

	check_vector_zero(matrix);

	return;
}
