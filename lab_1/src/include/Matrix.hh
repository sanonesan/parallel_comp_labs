#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

template <class M>
class Matrix {
   private:
	std::size_t _rows = 0;
	std::size_t _cols = 0;
	std::vector<M> _array = {};

   public:
	Matrix(std::size_t rows, std::size_t cols);
	Matrix(std::size_t n);

	std::size_t cols() { return this->_cols; };
	std::size_t rows() { return this->_rows; };

	std::vector<M>& operator()(const std::size_t i);
	M operator()(const std::size_t i, const std::size_t j);

	void print();
};

template <class M>
inline Matrix<M>::Matrix(std::size_t rows, std::size_t cols) {
	this->_rows = rows;
	this->_cols = cols;
	this->_array.reserve(rows * cols);

	for (auto i = 0; i < rows * cols; ++i) this->_array.push_back((M)0.);
}

template <class M>
inline Matrix<M>::Matrix(std::size_t n) {
	this->_rows = n;
	this->_cols = n;
	this->_array.reserve(n * n);

	for (auto i = 0; i < n * n; ++i) this->_array.push_back((M)0.);
}

template <class M>
inline void Matrix<M>::print() {
	std::cout << "[\n";
	for (std::size_t i = 0; i < this->_rows; ++i) {
		for (std::size_t j = 0; j < this->_cols; ++j) {
			std::cout << this->_array[i + this->_cols * j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "]\n";
}

template <class M>
inline std::vector<M>& Matrix<M>::operator()(const std::size_t i) {
	std::vector<M> line_i = {};
	line_i.reserve(this->_cols);

	for (std::size_t j = 0; j < this->_cols; ++j) {
		line_i.push_back(this->_array[i + this->_cols * j]);
	}

	return line_i;
}

template <class M>
inline M Matrix<M>::operator()(const std::size_t i, const std::size_t j) {
	return this->_array[i + this->_cols * j];
}
