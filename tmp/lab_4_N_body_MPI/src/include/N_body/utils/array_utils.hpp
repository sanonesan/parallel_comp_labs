#pragma once

#include <array>
#include <cmath>
#include <vector>

#include "../struct_Body.hpp"
// умножение вектора на число
template <typename T>
std::array<T, 3>& operator*=(std::array<T, 3>& A, T alpha) {
	for (std::size_t i = 0; i < A.size(); ++i) A[i] = A[i] * alpha;
	return A;
}

template <typename T>
std::array<T, 3> operator*(std::array<T, 3> A, T alpha) {
	for (std::size_t i = 0; i < A.size(); ++i) A[i] = A[i] * alpha;
	return A;
}

template <typename T>
std::array<T, 3> operator*(T alpha, std::array<T, 3> A) {
	return A * alpha;
}

// сложение векторов
template <typename T>
std::array<T, 3> operator+(const std::array<T, 3>& A,
						   const std::array<T, 3>& B) {
	std::array<T, 3> res(A);
	for (int i = 0; i < A.size(); ++i) res[i] += B[i];
	return res;
}


template <typename T>
std::array<T, 3>& operator+=(std::array<T, 3>& A, const std::array<T, 3>& B) {
	std::array<T, 3> res(A);
	for (int i = 0; i < A.size(); ++i) A[i] += B[i];
	return A;
}

// разность векторов
template <typename T>
std::array<T, 3> operator-(const std::array<T, 3>& A,
						   const std::array<T, 3>& B) {
	std::array<T, 3> res(A);
	for (int i = 0; i < A.size(); ++i) res[i] -= B[i];
	return res;
}

template <typename T>
std::array<T, 3>& operator-=(std::array<T, 3>& A, const std::array<T, 3>& B) {
	std::array<T, 3> res(A);
	for (int i = 0; i < A.size(); ++i) A[i] -= B[i];
	return A;
}

// норма
template <typename T>
T arr_norm(const std::array<T, 3>& A) {
	T sum = 0.0;
	for (int i = 0; i < A.size(); ++i) sum += A[i] * A[i];
	return sqrt(sum);
}
