#pragma once

#include <random>
#include <vector>

template <typename T>
void fill_martrix_with_random_numbers(std::vector<T>& matrix, std::size_t n,
									  int min_num = 1, int max_num = 50) {
	for (std::size_t i = 0; i < n * n; ++i)
		if ((rand() % (10 - 1 + 1) + 1) > 5)
			matrix.push_back(rand() % (max_num - min_num + 1) + 1);
		else
			matrix.push_back(-(rand() % (max_num - min_num + 1) + 1));

	return;
};
