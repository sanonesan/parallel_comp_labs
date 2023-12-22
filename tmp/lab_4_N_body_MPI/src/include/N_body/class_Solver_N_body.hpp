#pragma once
#include <mpi.h>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../../config.hpp"
#include "./schemes/ode_RK2_MPI.hpp"
#include "./struct_ode_system.hpp"


// inline double rand_double(double a, double b) {
// 	return (b - a) * rand() / RAND_MAX + a;
// }
//
// template <typename T>
// std::vector<Body<T>> generate_data(std::size_t N) {
// 	std::vector<Body<T>> res;
// 	res.reserve(N);
//
// 	Body<T> body{};
// 	for (std::size_t i = 0; i < N; ++i) {
// 		body.m = rand_double(1e6, 1e8);
// 		for (std::size_t j = 0; j < 3; ++j) {
// 			body.r[j] = rand_double(-1e2, 1e2);
// 			body.v[j] = rand_double(-1e1, 1e1);
// 		}
//
// 		res.push_back(body);
// 	}
// 	return res;
// }

template <typename T>
class Solver_N_body {
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
	T timer = 0.;
	std::size_t iter = 0;
	std::string file_name = "";
	std::string output_folder = "../output/";

	// Default constructor
	Solver_N_body() {
		this->tol = 1e-6;
		this->output_folder = "../output/";
		this->file_name = "";
	}

	// Alt constructor
	Solver_N_body(T tol, std::string output_folder, std::string file_name) {
		this->tol = tol;
		this->output_folder = output_folder;
		this->file_name = file_name;
	}


	void solve_parallel_MPI(ode_system<T>& sys, std::string OS = "PC") {
		if (this->notifications) {
			std::cout << file_name << ": solving... \t";
		}

		std::ofstream output_file;
		int id, num_procs;
		MPI_Comm_rank(MPI_COMM_WORLD, &id);
		MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

		if (id == 0) {
			if (OS == "PC") {
				output_file.open("../output/output_" + sys.name + "_" +
								 std::to_string(sys.tau) + "_np_" +
								 std::to_string(num_procs) + ".csv");
			}
			if (OS == "Server") {
				output_file.open(config_lab_4_MPI::PATH_output_folder +
								 "/output_" + sys.name + "_" +
								 std::to_string(sys.tau) + "_np_" +
								 std::to_string(num_procs) + ".csv");
			}
			// Init csv header
			output_file << "size;t_start;t_finish;tau;time_steps;"
						   "num_procs;time_taken;avg_time_for_step\n";
		}

		std::size_t size = sys.traj_init.size();
		if (num_procs > size) {
			throw "The number of processes is greater than the problem dimension";
		}

		std::size_t count_all = size / num_procs;
		std::size_t count_last = size - count_all * (num_procs - 1);
		int* recv_count = new int[num_procs];
		for (std::size_t i = 0; i < num_procs; ++i) {
			recv_count[i] = count_all;
		}
		recv_count[num_procs - 1] = count_last;

		int* offsets = new int[num_procs];
		for (std::size_t i = 0; i < num_procs; ++i) {
			offsets[i] = i * count_all;
		}


		MPI_Datatype tmp;
		const int count = 1;
		MPI_Aint offsets_type[count] = {offsetof(Body<T>, r)};
		int len[count] = {3};
		MPI_Datatype types[count] = {MPI_DOUBLE};
		MPI_Type_create_struct(count, len, offsets_type, types, &tmp);
		MPI_Type_commit(&tmp);

		MPI_Datatype MPI_BODY_R;
		MPI_Type_create_resized(tmp, offsetof(Body<T>, r), sizeof(Body<T>),
								&MPI_BODY_R);
		MPI_Type_commit(&MPI_BODY_R);

		if (sys.name == "4Body_test") {
			std::string test_path =
				"../output/results_" + std::to_string(sys.tau) + ".csv";
			ode_RK2_MPI(sys, iter, timer, MPI_BODY_R, recv_count, offsets, id,
						test_path);
		}
		if (sys.name == "4Body_random") {
			ode_RK2_MPI(sys, iter, timer, MPI_BODY_R, recv_count, offsets, id);
			// if (id == 0) {
			// 	auto data = generate_data<T>(sys.N_bodies);
			// 	std::ofstream out_rand;
			// 	out_rand.open("../output/rand_bodies.txt");
			// 	out_rand << sys.N_bodies << "\n";
			// 	for (auto& body : data) {
			// 		out_rand << body.m;
			// 		out_rand << "\t" << body.r[0] << "\t" << body.r[1] << "\t"
			// 				 << body.r[2] << "\t";
			// 		out_rand << "\t" << body.v[0] << "\t" << body.v[1] << "\t"
			// 				 << body.v[2] << "\n";
			// 	}
			// 	out_rand.close();
			// }
			// MPI_Bcast(sys.traj_init.data(), sys.N_bodies * (1 + 3 + 3),
			// 		  MPI_DOUBLE, 0, MPI_COMM_WORLD);

			// ode_RK2_MPI(sys, iter, timer, MPI_BODY_R, recv_count, offsets,
			// id,
			// 			"");
		}

		MPI_Type_free(&tmp);
		MPI_Type_free(&MPI_BODY_R);
		MPI_Barrier(MPI_COMM_WORLD);
		if (id == 0) {
			output_file << size << ";" << sys.t_start << ";" << sys.t_final
						<< ";" << sys.tau << ";" << sys.n_time_steps << ";"
						<< num_procs << ";" << timer << ";"
						<< timer / sys.n_time_steps << "\n";
		}
		output_file.close();
		if (this->notifications) {
			std::cout << "  Done!\n";
		}
	};


	/**
	 * Output results into files
	 */
	void output_eq(ode_system<T>& eq) {
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

		if (this->notifications) {
			std::cout << "  Done!\n";
		}
	};
};
