#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>


template <class T>
class Helmholtz_equation {
   public:

	std::string _name;

	// Space Rectangle G = [_x1_0, _x1_L1] x [_x2_0, _x2_L2]
	T _x1_0 = 0.;
	T _x1_L1 = 1.;

	T _x2_0 = 0.;
	T _x2_L2 = 1.;

	T _h = 0.1;
	T _h_2 = 0.01;

	T _k = 0.5;
	T _coef = 4. + _h * _h * _k * _k;


	// Dirichlete = 0 / Neuman = 1 boundary conditions pair = {condition type,
	// condition function}
	std::pair<std::size_t, std::function<T(const T x, const T y)>>
		_left_boundary_condition;
	std::pair<std::size_t, std::function<T(const T x, const T y)>>
		_right_boundary_condition;
	std::pair<std::size_t, std::function<T(const T x, const T y)>>
		_lower_boundary_condition;
	std::pair<std::size_t, std::function<T(const T x, const T y)>>
		_upper_boundary_condition;

	std::function<T(const T x1, const T x2)> _f;


	std::vector<T> x1, x2, res;

	/**
	 * Class constructor with default test Helmholtz_equation
	 */
	Helmholtz_equation() {
		this->_x1_0 = 0.;
		this->_x1_L1 = 1.;
		this->_x2_0 = 0.;
		this->_x2_L2 = 1.;

		this->_h = 0.1;
		this->_h_2 = this->_h * this->_h;

		auto f = [this](T x, T y) {
			return 2 * sin(M_PI * y) +
				   this->_k * this->_k * (1 - x) * x * sin(M_PI * y) +
				   M_PI * M_PI * (1 - x) * x * sin(M_PI * y);
		};
		this->_f = f;
		this->_left_boundary_condition.first = 0;
		this->_right_boundary_condition.first = 0;
		this->_upper_boundary_condition.first = 0;
		this->_lower_boundary_condition.first = 0;

		this->_left_boundary_condition.second = [](const T x, const T y) -> T {
			return 0;
		};
		this->_right_boundary_condition.second = [](const T x, const T y) -> T {
			return 0;
		};
		this->_upper_boundary_condition.second = [](const T x, const T y) -> T {
			return 0;
		};
		this->_lower_boundary_condition.second = [](const T x, const T y) -> T {
			return 0;
		};
	}

	/*
	 * May be realize later
	 */


	// std::string name(){};
	//
	// // Space Rectangle G = [_x1_0, _x1_L1] x [_x2_0, _x2_L2]
	// std::pair<T, T> x1_bounds() {
	// 	return std::pair<T, T>{this->_x1_0, this->_x1_L1};
	// };
	//
	// std::pair<T, T> x2_bounds() {
	// 	return std::pair<T, T>{this->_x2_0, this->_x2_L2};
	// };

	// T h() { return this->_h; }
	//
	// T h_2() { return this->_h_2; }
	//
	// T k() { return this->_k; }
	// T coef() { return this->_coef; }
	// // Dirichlete = 0 / Neuman = 1 boundary conditions pair = {condition
	// type,
	// // condition function}
	// std::pair<std::size_t, std::function<T(const T x, const T y)>>
	// left_boundary_condition() {
	// 	return this->_left_boundary_condition;
	// };
	// std::pair<std::size_t, std::function<T(const T x, const T y)>>
	// right_boundary_condition() {
	// 	return this->_right_boundary_condition;
	// };
	// std::pair<std::size_t, std::function<T(const T x, const T y)>>
	// lower_boundary_condition() {
	// 	return this->_lower_boundary_condition;
	// };
	// std::pair<std::size_t, std::function<T(const T x, const T y)>>
	// upper_boundary_condition() {
	// 	return this->_upper_boundary_condition;
	// };
	//
	// std::function<T(const T x1, const T x2)> f() { return this->_f; };
};
