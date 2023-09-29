#pragma once

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>


template <typename T>
void check_vector_zero(std::vector<T>& vec) {
	const T _eps = 1e-14;
	for (std::size_t i = 0; i < vec.size(); ++i) {
		if (fabs(vec[i]) < _eps) {
			vec[i] = 0.;
			vec[i] *= (1 / vec[i] > 0) ? (1) : (-1);
		}
	}
}

/**
 *   Prstd::size_t 1d array in Matrix form
 */
template <typename T>
void print_matrix(const std::vector<T>& matrix, const std::size_t n,
				  const std::size_t m = 0) {
	std::cout << "[\n";

	std::size_t v = n;
	if (m != 0) v = m;

	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < v; ++j) {
			std::cout << std::setw(10) << matrix[i * n + j] << ", ";
		}
		std::cout << "\n";
	}
	std::cout << "]\n";

	return;
}

/**
 *
 * Square matrices multiplication
 * RESULT = A * B
 *
 */
template <typename T>
std::vector<T> matrix_mult(const std::vector<T>& A, const std::vector<T>& B,
						   const std::size_t n) {
	std::vector<T> Result;
	Result.reserve(n * n);
	Result.resize(n * n);
	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < n; ++j) {
			Result[i * n + j] = 0.;
			for (std::size_t k = 0; k < n; ++k) {
				Result[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
	return Result;
}


/**
 *
 * Get Lower and Upper matrices of given matrix
 *
 */
template <typename T>
void get_LU_matrices(const std::vector<T>& matrix, std::vector<T>& L,
					 std::vector<T>& U, const std::size_t n) {
	L.clear();
	U.clear();
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


/**
 *
 *
 * Find Iverse Matrix
 *
 * via QR decomposion
 *
 */
template <typename T>
std::vector<T> inverse_Matrix(const std::vector<T>& A, const std::size_t n) {
	T eps = 1e-14;
	auto reverse_course = [n](const std::vector<T>& A, const std::vector<T>& b,
							  std::vector<T>& solution) -> int {
		solution = b;
		solution[n - 1] /= A[(n - 1) * n + n - 1];

		for (std::size_t i = 0; i <= n - 2; ++i) {
			for (std::size_t j = n - 2 - i + 1; j < n; ++j) {
				solution[n - 2 - i] -= solution[j] * A[(n - 2 - i) * n + j];
			}
			solution[n - 2 - i] /= A[(n - 2 - i) * n + n - 2 - i];
		}

		check_vector_zero(solution);

		return 0;
	};

	auto coefs = [eps, n](const std::size_t& k, const std::size_t& l, T& c,
						  T& s, const std::vector<T>& A) -> int {
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

		check_vector_zero(Q);
		check_vector_zero(R);

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

		reverse_course(R, tmp, tmp_solution);

		for (std::size_t j = 0; j < n; ++j) {
			InvMatrix[j * n + i] = tmp_solution[j];
		}
	}

	check_vector_zero(InvMatrix);

	return InvMatrix;
}

template <typename T>
void multiplyBlockPart(T* a, T* b, T* c, std::size_t size) {
	const std::size_t bs = 64;

	T abl[bs * bs], bbl[bs * bs], cbl[bs * bs];

	for (std::size_t bi = 0; bi < size; bi += bs)
		for (std::size_t bj = 0; bj < size; bj += bs) {
			// 1. Обнулить очередную матрицу C для накопления результата
			// умножения блочной строки на столбец
			for (std::size_t p = 0; p < bs; ++p)
				for (std::size_t q = 0; q < bs; ++q) cbl[p * bs + q] = 0.0;

			for (std::size_t bk = 0; bk < size; bk += bs) {
				// 2. Извлечь из глобальных матриц A и B очередные блоки (bs x
				// bs) для перемножения
				for (std::size_t p = 0; p < bs; ++p)
					for (std::size_t q = 0; q < bs; ++q) {
						abl[p * bs + q] = a[(bi + p) * size + (bk + q)];
						bbl[p * bs + q] = b[(bk + p) * size + (bj + q)];
					}

				// 3. Перемножить блоки bs x bs и добавить в блочную матрицу C
				for (size_t i = 0; i < bs; ++i)
					for (size_t k = 0; k < bs; ++k)
						for (size_t j = 0; j < bs; ++j)
							cbl[i * bs + j] +=
								abl[i * bs + k] * bbl[k * bs + j];
			}

			// 4. Переписать результат блочного умножения в глобальную матрицу
			for (std::size_t p = 0; p < bs; ++p)
				for (std::size_t q = 0; q < bs; ++q)
					c[(bi + p) * size + (bj + q)] = cbl[p * bs + q];
		}
}

template <typename T>
void Strassen(T* A, T* B, T* C, std::size_t size) {
	std::size_t N = 512;  // Размер самого маленького блока

	if (size <= N)
		multiplyBlockPart(A, B, C, size);
	else {
		std::size_t n = (std::size_t)(size / 2);

		T* P = new T[7 * n * n];

		T* a = new T[n * n];
		T* b = new T[n * n];

		T *P1, *P2, *P3, *P4, *P5, *P6, *P7;

		P1 = P;
		P2 = P + n * n;
		P3 = P + 2 * n * n;
		P4 = P + 3 * n * n;
		P5 = P + 4 * n * n;
		P6 = P + 5 * n * n;
		P7 = P + 6 * n * n;

		for (std::size_t i = 0; i < n; i++)
			for (std::size_t j = 0; j < n; j++) {
				a[i * n + j] = A[i * size + j + n] - A[(i + n) * size + j + n];
				b[i * n + j] =
					B[(i + n) * size + j] + B[(i + n) * size + j + n];
			}

		Strassen(a, b, P1, n);

		for (std::size_t i = 0; i < n; i++)
			for (std::size_t j = 0; j < n; j++) {
				a[i * n + j] = A[i * size + j] + A[(i + n) * size + j + n];
				b[i * n + j] = B[i * size + j] + B[(i + n) * size + j + n];
			}

		Strassen(a, b, P2, n);

		for (std::size_t i = 0; i < n; i++)
			for (std::size_t j = 0; j < n; j++) {
				a[i * n + j] = A[i * size + j] - A[(i + n) * size + j];
				b[i * n + j] = B[i * size + j] + B[i * size + j + n];
			}

		Strassen(a, b, P3, n);

		for (std::size_t i = 0; i < n; i++)
			for (std::size_t j = 0; j < n; j++) {
				a[i * n + j] = A[i * size + j] + A[i * size + j + n];
				b[i * n + j] = B[(i + n) * size + j + n];
			}

		Strassen(a, b, P4, n);

		for (std::size_t i = 0; i < n; i++)
			for (std::size_t j = 0; j < n; j++) {
				a[i * n + j] = A[i * size + j];
				b[i * n + j] = B[i * size + j + n] - B[(i + n) * size + j + n];
			}

		Strassen(a, b, P5, n);

		for (std::size_t i = 0; i < n; i++)
			for (std::size_t j = 0; j < n; j++) {
				a[i * n + j] = A[(i + n) * size + j + n];
				b[i * n + j] = B[(i + n) * size + j] - B[i * size + j];
			}

		Strassen(a, b, P6, n);

		for (std::size_t i = 0; i < n; i++)
			for (std::size_t j = 0; j < n; j++) {
				a[i * n + j] =
					A[(i + n) * size + j] + A[(i + n) * size + j + n];
				b[i * n + j] = B[i * size + j];
			}

		Strassen(a, b, P7, n);

		// Собираем C
		for (std::size_t i = 0; i < n; i++)
			for (std::size_t j = 0; j < n; j++) {
				C[i * size + j] = P1[i * n + j] + P2[i * n + j] -
								  P4[i * n + j] + P6[i * n + j];
				C[i * size + j + n] = P4[i * n + j] + P5[i * n + j];
				C[(i + n) * size + j] = P6[i * n + j] + P7[i * n + j];
				C[(i + n) * size + j + n] = P2[i * n + j] - P3[i * n + j] +
											P5[i * n + j] - P7[i * n + j];
			}

		delete[] P;
		delete[] a;
		delete[] b;
	}
}
