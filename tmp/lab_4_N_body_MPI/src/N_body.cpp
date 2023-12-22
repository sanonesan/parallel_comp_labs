#include <mpi.h>

#include <cstddef>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "./config.hpp"
#include "./include/N_body/class_Solver_N_body.hpp"
#include "./include/N_body/struct_Body.hpp"
#include "./include/N_body/struct_ode_system.hpp"
#include "./include/N_body/utils/norm_difference.hpp"

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
int main(int argc, char** argv) {
	typedef double T;

	double t1, t2;
	std::string communication_type, method_type;
	bool OUTPUT_SOLUTION = false;


	ode_system<T> sys;

	// auto data = generate_data<T>(8000);
	// std::ofstream out_rand;
	// out_rand.open("../output/rand_bodies.txt");
	// out_rand << 8000 << "\n";
	// for (auto& body : data) {
	// 	out_rand << body.m;
	// 	out_rand << "\t" << body.r[0] << "\t" << body.r[1] << "\t" << body.r[2]
	// 			 << "\t";
	// 	out_rand << "\t" << body.v[0] << "\t" << body.v[1] << "\t" << body.v[2]
	// 			 << "\n";
	// }
	// out_rand.close();

	Solver_N_body<T> solver;
	solver.tol = 1e-6;
	solver.output_folder = solver.output_folder =
		config_lab_4_MPI::PATH_output_folder + "/solution";
	solver.notifications = false;

	MPI_Init(&argc, &argv);

	// MPI_Barrier(MPI_COMM_WORLD);
	// sys.ode_system_default_test(0.1);
	// solver.solve_parallel_MPI(sys);
	//
	// MPI_Barrier(MPI_COMM_WORLD);
	// sys.ode_system_default_test(0.05);
	// solver.solve_parallel_MPI(sys);
	//
	// MPI_Barrier(MPI_COMM_WORLD);
	// sys.ode_system_default_test(0.025);
	// solver.solve_parallel_MPI(sys);
	//
	MPI_Barrier(MPI_COMM_WORLD);
	sys.ode_system_random_data();
	solver.solve_parallel_MPI(sys, "PC");

	MPI_Finalize();
	return 0;
}
