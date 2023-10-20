#pragma once

#include <cstddef>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "../Matrix_functions.hpp"
#include "./class_Helmholtz.hpp"

template <typename T>
T norm_difference(const std::vector<T>& A, const std::vector<T>& B,
				  const std::size_t& rows, const std::size_t& cols,
				  const T& eps = 1e-15) {
	T norm = 0.;
	T tmp = 0.;

	for (std::size_t i = 0; i < rows; ++i) {
		tmp = 0.;
		for (std::size_t j = 0; j < cols; ++j) {
			tmp += fabs(A[i * cols + j] - B[i * cols + j]);
		}
		if (norm < tmp) norm = tmp;
	}
	if (fabs(norm) < eps && norm != 0.) {
		norm = 0.;
		if (1 / norm < 0) norm *= -1;
	}
	return norm;
}

template <typename T>
void Jacobi(Helmholtz_equation<T>& eq, std::size_t& iter, const T& eps = 1e-6) {
	std::vector<T> u0(eq.res);
	u0.shrink_to_fit();
	iter = 1;
	std::size_t n1 = eq.x2.size(), n2 = eq.x1.size();

	/**
	 * Main body of Jacobi method
	 */
	do {
		if (iter > 1) eq.res.swap(u0);
		for (std::size_t i = 1; i < n1 - 1; ++i)
			for (std::size_t j = 1; j < n2 - 1; ++j)
				eq.res[i * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[i], eq.x2[j]) + u0[i * n2 + j - 1] +
					 u0[i * n2 + j + 1] + u0[(i - 1) * n2 + j] +
					 u0[(i + 1) * n2 + j]) /
					eq._coef;
		++iter;
	} while (norm_difference(u0, eq.res, n1, n2, eps) > eps);

	return;
}


template <typename T>
void Zeidel(Helmholtz_equation<T>& eq, std::size_t iter, const T& eps = 1e-6) {
	std::vector<T> u0(eq.res);
	u0.shrink_to_fit();
	iter = 1;
	std::size_t n1 = eq.x2.size(), n2 = eq.x1.size();

	/**
	 * Main body of Zeidel method
	 */
	do {
		if (iter > 1) eq.res.swap(u0);
		for (std::size_t i = 1; i < n1 - 1; ++i)
			for (std::size_t j = 1; j < n2 - 1; ++j)
				eq.res[i * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[i], eq.x2[j]) +
					 eq.res[i * n2 + j - 1] + u0[i * n2 + j + 1] +
					 eq.res[(i - 1) * n2 + j] + u0[(i + 1) * n2 + j]) /
					eq._coef;
		++iter;
	} while (norm_difference(u0, eq.res, n1, n2, eps) > eps);

	return;
}


template <typename T>
void Zeidel_RB(Helmholtz_equation<T>& eq, std::size_t iter,
			   const T& eps = 1e-6) {
	std::vector<T> u0(eq.res);
	u0.shrink_to_fit();
	iter = 1;
	std::size_t n1 = eq.x2.size(), n2 = eq.x1.size();

	/**
	 * Main body of Zeidel_RB method
	 */
	do {
		if (iter > 1) eq.res.swap(u0);

		// Go through REDS
		for (std::size_t i = 1; i < n1 - 1; ++i)
			for (std::size_t j = 1 + (i - 1) % 2; j < n2 - 1; j += 2)
				eq.res[i * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[i], eq.x2[j]) + u0[i * n2 + j - 1] +
					 u0[i * n2 + j + 1] + u0[(i - 1) * n2 + j] +
					 u0[(i + 1) * n2 + j]) /
					eq._coef;

		// Go through BLACKS
		for (std::size_t i = 1; i < n1 - 1; ++i)
			for (std::size_t j = 1 + i % 2; j < n2 - 1; j += 2)
				eq.res[i * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[i], eq.x2[j]) +
					 eq.res[i * n2 + j - 1] + eq.res[i * n2 + j + 1] +
					 eq.res[(i - 1) * n2 + j] + eq.res[(i + 1) * n2 + j]) /
					eq._coef;
		++iter;
	} while (norm_difference(u0, eq.res, n1, n2, eps) > eps);

	return;
}


template <typename T>
void solve_Helmoltz_eq_sequential(Helmholtz_equation<T>& eq, std::size_t& iter,
								  const std::string method = "Zeidel_RB",
								  const T& eps = 1e-6) {
	if (method == "Jacobi") {
		Jacobi(eq, iter, eps);
		return;
	} else if (method == "Zeidel") {
		Zeidel(eq, iter, eps);
		return;
	} else if (method == "Zeidel_RB") {
		return Zeidel_RB(eq, iter, eps);
	}

	return;
}
