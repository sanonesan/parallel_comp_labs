#pragma once

#include <array>

template <typename T>
struct Body {
	T m;
	std::array<T, 3> r;
	std::array<T, 3> v;
};
