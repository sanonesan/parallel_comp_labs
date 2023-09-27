#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

template <typename T>
void lu_seq_decomp(std::vector<T>& matrix, std::size_t n) {
	for (std::size_t i = 0; i < n - 1; ++i) {
		for (std::size_t j = i + 1; j < n; ++j) {
			matrix[j * n + i] = matrix[j * n + i] / matrix[i * n + i];
		}
		for (std::size_t j = i + 1; j < n; ++j) {
			for (std::size_t k = i + 1; k < n; ++k) {
				matrix[j * n + k] =
					matrix[j * n + k] - matrix[j * n + i] * matrix[i * n + k];
			}
		}
	}
	return;
}
