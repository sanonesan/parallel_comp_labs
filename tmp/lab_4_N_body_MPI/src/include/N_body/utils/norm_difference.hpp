#pragma once

#include <math.h>

#include <cmath>
#include <string>
#include <vector>

template <typename T>
T norm_difference(const std::vector<T>& A, const std::vector<T>& B,
				  const std::size_t& rows, const std::size_t& cols,
				  const std::string norm_type = "inf", const T& eps = 1e-15) {
	T norm = 0.;
	T tmp = 0.;

	if (norm_type == "inf") {
		for (std::size_t i = 0; i < rows; ++i) {
			tmp = 0.;
			for (std::size_t j = 0; j < cols; ++j) {
				tmp += fabs(A[i * cols + j] - B[i * cols + j]);
			}
			if (norm < tmp) norm = tmp;
		}
	} else if (norm_type == "2") {
		for (std::size_t i = 0; i < A.size(); ++i) {
			tmp = A[i] - B[i];
			norm += tmp * tmp;
		}
		norm = sqrt(norm);
	}
	if (fabs(norm) < eps && norm != 0.) {
		norm = 0.;
		if (1 / norm < 0) norm *= -1;
	}
	return norm;
}
