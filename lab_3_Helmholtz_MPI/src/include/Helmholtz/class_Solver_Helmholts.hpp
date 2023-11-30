#pragma once
#include <mpi.h>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "./choose_scheme.hpp"
#include "./class_Helmholtz.hpp"


template <class T>
class Solver_Helmoltz_eq {
   private:

	void check_folder(const std::string& str) {
		std::ifstream dir_stream(str.c_str());

		if (!dir_stream) {
			std::cout << "Folder created!\n"
					  << "path:\t" << str << "\n\n";
			const int dir_err =
				mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
			if (dir_err == -1) {
				std::cout << ("Error creating directory!\n");
				exit(1);
			}
		}
	}

   public:

	T tol = 1e-6;
	bool notifications = false;
	std::size_t iter = 0;
	std::string file_name = "";
	std::string output_folder = "../output/";

	// Default constructor
	Solver_Helmoltz_eq() {
		this->tol = 1e-6;
		this->output_folder = "../output/";
		this->file_name = "";
	}

	// Alt constructor
	Solver_Helmoltz_eq(T tol, std::string output_folder,
					   std::string file_name) {
		this->tol = tol;
		this->output_folder = output_folder;
		this->file_name = file_name;
	}

	/*
	 * Initialize equation:
	 *  - reserve memory for answer
	 *  - reserve memory for grid
	 *  - set boundary conditions
	 */
	void compute(Helmholtz_equation<T>& eq) {
		if (this->notifications) {
			std::cout << file_name << ": computation...\t";
		}
		std::size_t n1 = std::size_t(floor((eq._x1_L1 - eq._x1_0) / eq._h) + 1);
		std::size_t n2 = std::size_t(floor((eq._x2_L2 - eq._x2_0) / eq._h) + 1);

		eq.x1.resize(n2);
		eq.x2.resize(n1);

		eq.x1[0] = eq._x1_0;
		for (std::size_t i = 1; i < n2 - 1; ++i) {
			eq.x1[i] = eq.x1[i - 1] + eq._h;
		}
		eq.x1[n2 - 1] = eq._x1_L1;

		eq.x2[0] = eq._x2_0;
		for (std::size_t i = 1; i < n1 - 1; ++i) {
			eq.x2[i] = eq.x2[i - 1] + eq._h;
		}
		eq.x2[n1 - 1] = eq._x2_L2;

		eq.res = std::vector<T>(n1 * n2, 0.0);
		// Set initial_conditions
		for (std::size_t i = 0; i < n2; ++i) {
			eq.res[i] = eq._lower_boundary_condition.second(eq.x1[i], eq.x2[0]);
			eq.res[(n1 - 1) * n2 + i] =
				eq._upper_boundary_condition.second(eq.x1[i], eq.x2[n1 - 1]);
		}
		for (std::size_t j = 0; j < n1; ++j) {
			eq.res[j * n2] =
				eq._left_boundary_condition.second(eq.x1[0], eq.x2[j]);
			eq.res[j * n2 + n1 - 1] =
				eq._right_boundary_condition.second(eq.x1[n2 - 1], eq.x2[j]);
		}
		eq.u0.resize(eq.res.size());
		eq.u0.assign(eq.res.begin(), eq.res.end());

		if (eq._solution.first == 1) {
			eq._solution_vec.reserve(n1 * n2);
			eq._solution_vec.resize(n1 * n2);
			for (std::size_t i = 0; i < n2; ++i) {
				for (std::size_t j = 0; j < n1; ++j) {
					eq._solution_vec[i * n2 + j] =
						eq._solution.second(eq.x1[i], eq.x2[j]);
				}
			}
		}

		if (this->notifications) {
			std::cout << "  Done!\n";
		}
	};


	/**
	 * Solve Helmholtz_equation sequentially
	 * with one of included methods:
	 * - Jacobi
	 * - Seidel
	 * - Seidel_RB
	 */
	void solve_sequential(Helmholtz_equation<T>& helmholts_eq,
						  const std::string method = "Jacobi") {
		if (this->notifications) {
			std::cout << file_name << ": solving... \t";
		}

		solve_Helmoltz_eq_sequential(helmholts_eq, this->iter, method,
									 this->tol);

		if (this->notifications) {
			std::cout << "  Done!\n";
		}
	};


	void solve_parallel_MPI(int id, int num_procs,
							Helmholtz_equation<T>& helmholts_eq,
							const std::string method = "Jacobi",
							const std::string communication_type = "SendRecv") {
		if (this->notifications) {
			std::cout << file_name << ": solving... \t";
		}


		solve_Helmoltz_eq_parallel_MPI(id, num_procs, helmholts_eq, this->iter,
									   method, communication_type, this->tol);

		if (this->notifications) {
			std::cout << "  Done!\n";
		}
	};


	/**
	 * Output results into files
	 */
	void output_eq(Helmholtz_equation<T>& eq) {
		this->check_folder(this->output_folder);
		std::string out_path;

		out_path = this->output_folder + "/" + this->file_name;

		if (this->notifications) {
			std::cout << file_name << ": output\t";
		}
		out_path += "_Helmoltz_eq_output";

		std::ofstream output_file;
		output_file << std::scientific;
		output_file << std::setprecision(8);

		output_file.open(out_path + "_x1.csv");
		if (!output_file.is_open()) {
			return;
		}
		output_file << "x1\n";
		for (std::size_t i = 0; i < eq.x1.size(); ++i) {
			output_file << eq.x1[i] << "\n";
		}
		output_file.close();


		output_file.open(out_path + "_x2.csv");
		if (!output_file.is_open()) {
			return;
		}
		output_file << "x2\n";
		for (std::size_t i = 0; i < eq.x2.size(); ++i) {
			output_file << eq.x2[i] << "\n";
		}
		output_file.close();


		output_file.open(out_path + "_res.csv");
		if (!output_file.is_open()) {
			return;
		}

		for (std::size_t j = 0; j < eq.x1.size(); ++j) {
			output_file << "u_" << j << ",";
		}
		output_file << "u_" << eq.x1.size() - 1 << "\n";

		for (std::size_t i = 0; i < eq.x2.size(); ++i) {
			output_file << "u_" << i << ",";
			for (std::size_t j = 0; j < eq.x1.size() - 1; ++j) {
				output_file << eq.res[i * eq.x1.size() + j] << ",";
			}
			output_file << eq.res[i * eq.x1.size() + eq.x1.size() - 1] << "\n";
		}
		output_file.close();

		if (this->notifications) {
			std::cout << "  Done!\n";
		}
	};
};
