#include <mpi.h>

#include <cstddef>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "./config.hpp"
#include "./include/Helmholtz/class_Helmholtz.hpp"
#include "./include/Helmholtz/class_Solver_Helmholts.hpp"
#include "./include/Helmholtz/helpers/norm_difference_parallel.hpp"

int main(int argc, char** argv) {
	typedef double T;

	double t1, t2;
	std::string algorithm_type, method_type;
	bool OUTPUT_SOLUTION = false;


	Helmholtz_equation<T> eq;
	eq.update_h(0.001);
	eq.update_k(1000.0);


	std::ofstream output_file;
	// output_file.open("../output/output_h_001.csv");
	// std::ofstream output_file(config_lab_2_OMP::PATH_output_folder +
	// "/output_h_01.csv");

	// Init csv header
	// output_file <<
	// "h_lenght,eps,algorithm_type,num_threads,method_type,n_of_"
	// 			   "iter,error_in_"
	// 			   "norm_inf,time_taken\n";

	/*
	 * Output naming format:
	 * A_h_eq_B_out
	 * A = s (Sequential) / p (Parallel)
	 * h_eq = helmholts_equation
	 * B = j (Jacobi) / s (Seidel) / zrb (Zeidel Red-Black iterationd)
	 */
	Solver_Helmoltz_eq<T> solver;
	solver.tol = 1e-6;
	solver.output_folder = "../output/solution";
	// solver.output_folder = config_lab_2_OMP::PATH_output_folder +
	// "/solution";
	solver.notifications = false;

	int id, num_procs;
	// Инициализация подсистемы MPI
	MPI_Init(&argc, &argv);
	// Получить размер коммуникатора MPI_COMM_WORLD
	// (общее число процессов в рамках задачи)
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	// Получить номер текущего процесса в рамках
	// коммуникатора MPI_COMM_WORLD
	MPI_Comm_rank(MPI_COMM_WORLD, &id);


	/**
	 * Jobs' section
	 */
	// solver.compute(eq);
	// MPI_Barrier(MPI_COMM_WORLD);
	// t1 = -MPI_Wtime();
	// solver.solve_parallel_MPI(id, num_procs, eq, "Jacobi", "Send_Recv");
	// t1 += MPI_Wtime();
	// MPI_Barrier(MPI_COMM_WORLD);
	// if (id == 0) {
	// 	std::cout << t1 << std::endl;
	// 	std::cout << norm_difference_parallel(eq._solution_vec, eq.res,
	// 										  eq.x1.size(), eq.x2.size())
	// 			  << std::endl;
	// }
	//
	// solver.compute(eq);
	// MPI_Barrier(MPI_COMM_WORLD);
	// t1 = -MPI_Wtime();
	// solver.solve_parallel_MPI(id, num_procs, eq, "Jacobi", "Sendrecv");
	// t1 += MPI_Wtime();
	// MPI_Barrier(MPI_COMM_WORLD);
	// if (id == 0) {
	// 	std::cout << t1 << std::endl;
	// 	std::cout << norm_difference_parallel(eq._solution_vec, eq.res,
	// 										  eq.x1.size(), eq.x2.size())
	// 			  << std::endl;
	// }
	//
	// solver.compute(eq);
	// MPI_Barrier(MPI_COMM_WORLD);
	// t1 = -MPI_Wtime();
	// solver.solve_parallel_MPI(id, num_procs, eq, "Jacobi", "SendI_RecvI");
	// t1 += MPI_Wtime();
	// MPI_Barrier(MPI_COMM_WORLD);
	// if (id == 0) {
	// 	std::cout << t1 << std::endl;
	// 	std::cout << norm_difference_parallel(eq._solution_vec, eq.res,
	// 										  eq.x1.size(), eq.x2.size())
	// 			  << std::endl;
	// }
	//
	//
	solver.compute(eq);
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = -MPI_Wtime();
	solver.solve_parallel_MPI(id, num_procs, eq, "Seidel_RB", "Send_Recv");
	t1 += MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	if (id == 0) {
		std::cout << t1 << std::endl;
		std::cout << norm_difference_parallel(eq._solution_vec, eq.res,
											  eq.x1.size(), eq.x2.size())
				  << std::endl;
	}

	solver.compute(eq);
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = -MPI_Wtime();
	solver.solve_parallel_MPI(id, num_procs, eq, "Seidel_RB", "Sendrecv");
	t1 += MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	if (id == 0) {
		std::cout << t1 << std::endl;
		std::cout << norm_difference_parallel(eq._solution_vec, eq.res,
											  eq.x1.size(), eq.x2.size())
				  << std::endl;
	}

	solver.compute(eq);
	MPI_Barrier(MPI_COMM_WORLD);
	t1 = -MPI_Wtime();
	solver.solve_parallel_MPI(id, num_procs, eq, "Seidel_RB", "ISend_IRecv");
	t1 += MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	if (id == 0) {
		std::cout << t1 << std::endl;
		std::cout << norm_difference_parallel(eq._solution_vec, eq.res,
											  eq.x1.size(), eq.x2.size())
				  << std::endl;
	}

	MPI_Finalize();
	output_file.close();
	return 0;
}
