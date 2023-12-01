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
	std::string communication_type, method_type;
	bool OUTPUT_SOLUTION = false;


	Helmholtz_equation<T> eq;
	eq.update_h(0.001);
	eq.update_k(1000.0);


	std::ofstream output_file;

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
	// solver.output_folder = config_lab_3_OMP::PATH_output_folder +
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

	if (id == 0) {
		// output_file.open("../output/output_h_001_np_" +
		// 				 std::to_string(num_procs) + ".csv");
		output_file.open(config_lab_3_MPI::PATH_output_folder +
						 "/output_h_001_nn_1_np_" + std::to_string(num_procs) +
						 ".csv");

		// Init csv header
		output_file
			<< "h_lenght;eps;num_procs;method_type;communication_type;n_of_"
			   "iter;error_in_"
			   "norm_inf;time_taken\n";
	}
	MPI_Barrier(MPI_COMM_WORLD);

	/**
	 * Jobs' section
	 */
	method_type = "Jacobi";
	communication_type = "Send_Recv";
	solver.compute(eq);
	t1 = -MPI_Wtime();
	solver.solve_parallel_MPI(id, num_procs, eq, method_type,
							  communication_type);
	t1 += MPI_Wtime();
	if (id == 0) {
		output_file << eq._h << ";" << solver.tol << ";" << communication_type
					<< ";" << num_procs << ";" << method_type << ";"
					<< solver.iter << ";"
					<< norm_difference_parallel(eq._solution_vec, eq.res,
												eq.x1.size(), eq.x2.size())
					<< ";" << t1 << "\n";
	}

	method_type = "Jacobi";
	communication_type = "Sendrecv";
	solver.compute(eq);
	t1 = -MPI_Wtime();
	solver.solve_parallel_MPI(id, num_procs, eq, method_type,
							  communication_type);
	t1 += MPI_Wtime();
	if (id == 0) {
		output_file << eq._h << ";" << solver.tol << ";" << communication_type
					<< ";" << num_procs << ";" << method_type << ";"
					<< solver.iter << ";"
					<< norm_difference_parallel(eq._solution_vec, eq.res,
												eq.x1.size(), eq.x2.size())
					<< ";" << t1 << "\n";
	}

	method_type = "Jacobi";
	communication_type = "ISend_IRecv";
	solver.compute(eq);
	t1 = -MPI_Wtime();
	solver.solve_parallel_MPI(id, num_procs, eq, method_type,
							  communication_type);
	t1 += MPI_Wtime();
	if (id == 0) {
		output_file << eq._h << ";" << solver.tol << ";" << communication_type
					<< ";" << num_procs << ";" << method_type << ";"
					<< solver.iter << ";"
					<< norm_difference_parallel(eq._solution_vec, eq.res,
												eq.x1.size(), eq.x2.size())
					<< ";" << t1 << "\n";
	}

	method_type = "Seidel_RB";
	communication_type = "Send_Recv";
	solver.compute(eq);
	t1 = -MPI_Wtime();
	solver.solve_parallel_MPI(id, num_procs, eq, method_type,
							  communication_type);
	t1 += MPI_Wtime();
	if (id == 0) {
		output_file << eq._h << ";" << solver.tol << ";" << communication_type
					<< ";" << num_procs << ";" << method_type << ";"
					<< solver.iter << ";"
					<< norm_difference_parallel(eq._solution_vec, eq.res,
												eq.x1.size(), eq.x2.size())
					<< ";" << t1 << "\n";
	}

	method_type = "Seidel_RB";
	communication_type = "Sendrecv";
	solver.compute(eq);
	t1 = -MPI_Wtime();
	solver.solve_parallel_MPI(id, num_procs, eq, method_type,
							  communication_type);
	t1 += MPI_Wtime();
	if (id == 0) {
		output_file << eq._h << ";" << solver.tol << ";" << communication_type
					<< ";" << num_procs << ";" << method_type << ";"
					<< solver.iter << ";"
					<< norm_difference_parallel(eq._solution_vec, eq.res,
												eq.x1.size(), eq.x2.size())
					<< ";" << t1 << "\n";
	}

	method_type = "Seidel_RB";
	communication_type = "ISend_IRecv";
	solver.compute(eq);
	t1 = -MPI_Wtime();
	solver.solve_parallel_MPI(id, num_procs, eq, method_type,
							  communication_type);
	t1 += MPI_Wtime();
	if (id == 0) {
		output_file << eq._h << ";" << solver.tol << ";" << communication_type
					<< ";" << num_procs << ";" << method_type << ";"
					<< solver.iter << ";"
					<< norm_difference_parallel(eq._solution_vec, eq.res,
												eq.x1.size(), eq.x2.size())
					<< ";" << t1 << "\n";
	}

	MPI_Finalize();
	output_file.close();
	return 0;
}
