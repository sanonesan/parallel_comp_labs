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

int main(int argc, char** argv) {
	typedef double T;

	double t1, t2;
	std::string communication_type, method_type;
	bool OUTPUT_SOLUTION = false;


	ode_system<T> sys;


	Solver_N_body<T> solver;
	solver.tol = 1e-6;
	solver.output_folder =
		"../output/solution";  // solver.output_folder =
							   // config_lab_4_OMP::PATH_output_folder +
	// "/solution";
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
	solver.solve_parallel_MPI(sys, "Server");

	MPI_Finalize();
	return 0;
}
