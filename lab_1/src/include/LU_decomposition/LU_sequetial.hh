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
	std::size_t step = b;

	auto algorithm_2_9 = [n](std::vector<T>& A, std::size_t ii,
							 std::size_t step) {
		//
		// A size is m * v
		//
		std::size_t m = n - ii;
		std::size_t v = step;

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


	bool flag = true;
	std::size_t sub_step = step;

	if (n % step != 0) {
		flag = false;
		sub_step += n % step;
	}

	std::vector<T> L22_block, identity_b_block, U23_block;
	L22_block.reserve(sub_step * sub_step);
	identity_b_block.reserve(sub_step * sub_step);
	U23_block.reserve(sub_step * sub_step);


	std::size_t i = 0;
	while (i < n - 1) {
		// for (std::size_t i = 0; i <= n - 1; i += step) {
		if (flag == false && i == 0) {
			// std::cout << "aaaa\n";
			step = sub_step;
			L22_block.resize(step * step);
			U23_block.resize(step * (n - step));
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
			// std::cout << "bbbb\n";
			flag = true;
			step = b;
			// std::cout << step << "\n";
			L22_block.resize(step * step);
			U23_block.resize(step * (n - step));
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
			// std::cout << "ccc\n";
			step = b;
			L22_block.resize(step * step);
			U23_block.resize(step * (n - step));
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

		// std::cout << i << flag << "\n";
		// std::cout << "pass0\n";
		algorithm_2_9(matrix, i, step);
		if (i == n - 1) break;
		// std::cout << "pass1\n";
		// print_matrix(matrix, n);
		L22_block.assign(identity_b_block.begin(), identity_b_block.end());
		// get L22_block
		for (std::size_t j = i + 1; j < i + step; ++j) {
			for (std::size_t k = i; k < j; ++k) {
				L22_block[(j - i) * step + (k - i)] = matrix[j * n + k];
			}
		}

		// std::cout << "pass2\n";
		// U23_block.resize(step * (n - step - i));
		for (std::size_t j = i; j < i + step; ++j) {
			for (std::size_t k = i + step; k < n; ++k) {
				// std::cout << j << " " << k << " " << matrix[j * n + k] <<
				// "\n";
				// std::cout << (j - i) << " " << k - i - step << " "
				// 		  << matrix[j * n + k] << "\n";
				U23_block[(j - i) * (n - step - i) + (k - i - step)] =
					matrix[j * n + k];
				// std::cout << (j - i) * (n - step - i) + (k - i - step) <<
				// "\n";
			}
		}
		// for (auto u = 0; u < U23_block.size(); ++u)
		// 	std::cout << U23_block[u] << "\n";
		// print_matrix(matrix, n);
		print_matrix(L22_block, step);
		//
		print_matrix(U23_block, step, n - step - i);
		// print_matrix(matrix, n);

		L22_block = inverse_Matrix(L22_block, step);
		print_matrix(L22_block, step);
		// std::cout << "pass4\n";
		// print_matrix(matrix, n);
		for (std::size_t l = i; l < i + step; ++l) {
			for (std::size_t j = i + step; j < n; ++j) {
				matrix[l * n + j] = 0.;
				for (std::size_t k = 0; k < step; ++k) {
					matrix[l * n + j] +=
						L22_block[(l - i) * step + k] *
						U23_block[k * (n - i - step) + (j - i - step)];
				}
			}
		}
		// std::cout << "pass5\n";
		// print_matrix(matrix, n);
		for (std::size_t l = i + step; l < n; ++l) {
			for (std::size_t j = i + step; j < n; ++j) {
				matrix[l * n + j] = 0.;
				for (std::size_t k = 0; k < n - step - i; ++k) {
					// std::cout << std::setw(9) << matrix[(i + b) * n + k]
					// << " * " << matrix[(i + k) * n + j] << "\n";
					matrix[l * n + j] +=
						matrix[l * n + k] * matrix[(i + k) * n + j];
				}
			}
		}
		// std::cout << "pass6\n";

		// print_matrix(L22_block, b);
		// print_matrix(U23_block, b, n - b - i);
		// print_matrix(matrix, n);

		i += step;


		// print_matrix(L22_block, b);
		// print_matrix(U23_block, b, n - b - i);
		// print_matrix(matrix, n);
	}


	return;
}
