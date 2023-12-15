#pragma once

#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

#include "../struct_Body.hpp"
#include "../struct_ode_system.hpp"

template <typename T>
std::vector<Body<T>> read_data(const std::string& path) {
	std::ifstream fout(path);

	if (!fout.is_open()) {
		std::cout << "file is't open" << std::endl;
		throw 1;
	}

	size_t N = 0;
	fout >> N;
	// std::cout << "N = " << N << std::endl;

	std::vector<Body<T>> res;
	res.reserve(N);

	Body<T> body{};
	for (size_t i = 0; i < N; ++i) {
		fout >> body.m >> body.r[0] >> body.r[1] >> body.r[2] >> body.v[0] >>
			body.v[1] >> body.v[2];

		res.push_back(body);
	}

	fout.close();

	return res;
}


template <typename T>
void write_bodies_coords(std::ostream& fout, double time,
						 const std::vector<Body<T>>& bodies) {
	fout << std::scientific;
	fout << std::setprecision(16);
	fout << time << ";";
	for (auto& body : bodies) {
		fout << body.r[0] << ";" << body.r[1] << ";" << body.r[2] << ";";
	}
	fout << std::endl;
}
