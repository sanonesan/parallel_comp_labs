#pragma once

#include <cmath>
#include <cstddef>
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
	return;
}


template <typename T>
std::vector<T> inverse_Matrix(std::vector<T>& A, std::size_t n) {
	T eps = 1e-14;
	auto reverse_course = [n](std::vector<T>& A, std::vector<T>& b,
							  std::vector<T>& solution) -> int {
		solution = b;
		solution[n - 1] /= A[(n - 1) * n + n - 1];

		for (std::size_t i = 0; i <= n - 2; ++i) {
			for (std::size_t j = n - 2 - i + 1; j < n; ++j) {
				solution[n - 2 - i] -= solution[j] * A[(n - 2 - i) * n + j];
			}
			solution[n - 2 - i] /= A[(n - 2 - i) * n + n - 2 - i];
		}

		// solution.check_vector_zero();

		return 0;
	};

	auto coefs = [eps, n](const std::size_t& k, const std::size_t& l, T& c,
						  T& s, std::vector<T>& A) -> int {
		if (k < n && l < n) {
			T temp = sqrt(pow(A[k * n + k], 2) + pow(A[l * n + k], 2));

			c = A[k * n + k] / temp;

			s = A[l * n + k] / temp;

			if (fabs(c) < eps) c = (1 / c > 0) ? c : c * (-1);
			if (fabs(s) < eps) s = (1 / s > 0) ? s : s * (-1);
		}

		return 0;
	};

	auto QR_decomposion_method = [coefs, reverse_course, eps, n](

									 const std::vector<T>& A, std::vector<T>& Q,
									 std::vector<T>& R) -> int {
		T c = 0., s = 0.;

		Q.clear();
		Q.resize(n * n);
		for (std::size_t i = 0; i < n; ++i) {
			for (std::size_t j = 0; j < n; ++j) {
				if (i != j)
					Q[i * n + j] = 0.;
				else
					Q[i * n + j] = 1.;
			}
		}
		// Q.make_matrix_identity(n);
		R = A;

		T a = 0., b = 0.;

		for (std::size_t i = 0; i < n - 1; ++i) {
			for (std::size_t j = i + 1; j < n; ++j) {
				coefs(i, j, c, s, R);

				for (std::size_t k = 0; k < n; ++k) {
					a = R[i * n + k];
					b = R[j * n + k];
					R[i * n + k] = (c * a + s * b);
					R[j * n + k] = (-s * a + c * b);

					a = Q[i * n + k];
					b = Q[j * n + k];
					Q[i * n + k] = (c * a + s * b);
					Q[j * n + k] = (-s * a + c * b);
				}
			}
		}

		// Q.check_matrix_zero();
		// R.check_matrix_zero();

		return 0;
	};

	std::vector<T> Q, R;

	QR_decomposion_method(A, Q, R);

	std::vector<T> InvMatrix(A);

	std::vector<T> b;
	std::vector<T> tmp;
	std::vector<T> tmp_solution;

	b.resize(n);
	tmp.resize(n);
	tmp_solution.resize(n);

	// for (std::size_t i = 0; i < n; ++i) {
	//     b[i] = 0.;
	//     tmp[i] = 0.;
	//     tmp_solution[i] = 0.;
	// }
	// Matrix<T> B(Q);

	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < n; ++j) {
			b[j] = (i == j) ? 1. : 0.;
			tmp[j] = 0.;
			tmp_solution[j] = 0.;
		}

		for (std::size_t j = 0; j < n; ++j) {
			for (std::size_t k = 0; k < n; ++k) {
				tmp[j] += Q[j * n + k] * b[k];
			}
		}
		// tmp = Q.dot(b);

		reverse_course(R, tmp, tmp_solution);

		for (std::size_t j = 0; j < n; ++j) {
			InvMatrix[j * n + i] = tmp_solution[j];
		}
	}

	// InvMatrix.transpose_this();
	// InvMatrix.check_matrix_zero();

	return InvMatrix;
}


/**
 * matrix --- матрица
 * n --- размерность матрицы
 * r --- блочный параметр (by default: r = 2)
 *
 */
template <typename T>
void block_lu_decomp(std::vector<T>& matrix, std::size_t n, std::size_t b = 2) {
	auto algorithm_2_9 = [n, b](std::vector<T>& A, std::size_t ii) {
		//
		// A size is m * v
		//
		std::size_t m = n - ii;
		std::size_t v = b - 1;

		for (std::size_t i = ii; i < ii + std::min(m - 1, v); ++i) {
			for (std::size_t j = i + 1; j < m; ++j) {
				A[j * m + i] = A[j * m + i] / A[i * m + i];
			}
			if (i < n) {
				for (std::size_t j = i + 1; j < m; ++j) {
					for (std::size_t k = i + 1; k < v; ++k) {
						A[j * m + k] -= A[j * m + i] * A[i * m + k];
					}
				}
			}
		}
	};

	std::vector<T> L22_block, identity_b_block;
	L22_block.resize(b * b);
	identity_b_block.resize(b * b);

	for (std::size_t i = 0; i < b; ++i) {
		for (std::size_t j = 0; j < b; ++j) {
			identity_b_block[i * b + j] = 0. ? (i != j) : 1.;
		}
	}

	for (std::size_t i = 0; i < n - 1; i += b) {
		algorithm_2_9(matrix, i);


		L22_block.assign(identity_b_block.begin(), identity_b_block.end());
		// get L22_block
		for (std::size_t j = i + 1; j < i + b; ++j) {
			for (std::size_t k = i; k < j; ++k) {
				L22_block[(j - i) * b + (k - i)] = matrix[j * n + k];
			}
		}

		L22_block = inverse_Matrix(L22_block, b);

		for (std::size_t j = i; j < i + b; ++j) {
			for (std::size_t k = i; k < n - i - b; ++k) {
			}
		}
	}


	return;
}
