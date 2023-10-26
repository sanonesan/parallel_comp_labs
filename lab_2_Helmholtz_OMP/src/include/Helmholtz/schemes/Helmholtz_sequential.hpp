#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "../Helmholtz_config/helmholtz_cfg.hpp"
#include "../class_Helmholtz.hpp"
#include "../helpers/norm_difference.hpp"

template <typename T>
void Jacobi(Helmholtz_equation<T>& eq, std::size_t& iter, const T& eps = 1e-6) {
	iter = 1;
	std::size_t n1 = eq.x2.size(), n2 = eq.x1.size();

	/**
	 * Main body of Jacobi method
	 */
	do {
		if (iter > 1) {
			eq.res.swap(eq.u0);
		};
		for (std::size_t i = 1; i < n1 - 1; ++i)
			for (std::size_t j = 1; j < n2 - 1; ++j)
				eq.res[i * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[i], eq.x2[j]) +
					 eq.u0[i * n2 + j - 1] + eq.u0[i * n2 + j + 1] +
					 eq.u0[(i - 1) * n2 + j] + eq.u0[(i + 1) * n2 + j]) /
					eq._coef;
		++iter;
	} while (norm_difference(eq.u0, eq.res, n1, n2,
							 config_Helmholtz_eq::NORM_TYPE, eps) > eps);

	return;
}


template <typename T>
void Seidel(Helmholtz_equation<T>& eq, std::size_t& iter, const T& eps = 1e-6) {
	iter = 1;
	std::size_t n1 = eq.x2.size(), n2 = eq.x1.size();

	/**
	 * Main body of Seidel method
	 */
	do {
		if (iter > 1) {
			eq.res.swap(eq.u0);
		};
		for (std::size_t i = 1; i < n1 - 1; ++i)
			for (std::size_t j = 1; j < n2 - 1; ++j)
				eq.res[i * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[i], eq.x2[j]) +
					 eq.res[i * n2 + j - 1] + eq.u0[i * n2 + j + 1] +
					 eq.res[(i - 1) * n2 + j] + eq.u0[(i + 1) * n2 + j]) /
					eq._coef;
		++iter;
	} while (norm_difference(eq.u0, eq.res, n1, n2,
							 config_Helmholtz_eq::NORM_TYPE, eps) > eps);

	return;
}


template <typename T>
void Seidel_RB(Helmholtz_equation<T>& eq, std::size_t& iter,
			   const T& eps = 1e-6) {
	iter = 1;
	std::size_t n1 = eq.x2.size(), n2 = eq.x1.size();

	/**
	 * Main body of Seidel_RB method
	 */
	do {
		if (iter > 1) {
			eq.res.swap(eq.u0);
		};

		// Go through REDS
		for (std::size_t i = 1; i < n1 - 1; ++i)
			for (std::size_t j = 1 + (i - 1) % 2; j < n2 - 1; j += 2)
				eq.res[i * n2 + j] =
					(eq._h_2 * eq._f(eq.x1[i], eq.x2[j]) +
					 eq.u0[i * n2 + j - 1] + eq.u0[i * n2 + j + 1] +
					 eq.u0[(i - 1) * n2 + j] + eq.u0[(i + 1) * n2 + j]) /
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
	} while (norm_difference(eq.u0, eq.res, n1, n2,
							 config_Helmholtz_eq::NORM_TYPE, eps) > eps);

	return;
}


template <typename T>
void solve_Helmoltz_eq_sequential(Helmholtz_equation<T>& eq, std::size_t& iter,
								  const std::string method = "Seidel_RB",
								  const T& eps = 1e-6) {
	if (method == "Jacobi") {
		Jacobi(eq, iter, eps);
		return;
	} else if (method == "Seidel") {
		Seidel(eq, iter, eps);
		return;
	} else if (method == "Seidel_RB") {
		Seidel_RB(eq, iter, eps);
		return;
	}

	return;
}
