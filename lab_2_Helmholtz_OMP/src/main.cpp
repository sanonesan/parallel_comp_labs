#include <omp.h>

#include <cstddef>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "./config.hpp"
#include "./include/Helmholtz/class_Helmholtz.hpp"
#include "./include/Helmholtz/class_Solver_Helmholts.hpp"
#include "./include/Matrix_functions.hpp"
#include "./include/Matrix_generator.hpp"


int main(int argc, char* argv[]) {
	typedef double T;

	std::ofstream output_file("../output/output_2048.txt");
	// std::ofstream output_file(config_lab_1::PATH_output_folder +
	// "/output_2048.txt");

	double t1, t2;

	Helmholtz_equation<T> eq;


	/*
	 * Output naming format:
	 * A_h_eq_B_out
	 * A = s (Sequential) / p (Parallel)
	 * h_eq = helmholts_equation
	 * B = j (Jacobi) / z (Zeidel) / zrb (Zeidel Red-Black iterationd)
	 */
	Solver_Helmoltz_eq<T> solver;
	solver.tol = 1e-6;
	solver.output_folder = "../output";

	// Solver_Helmoltz_eq<T> solver(1e-6, config_lab_1::PATH_output_folder,
	// 							 "Helmholtz_eq_out");
	//

	solver.notifications = true;
	solver.compute(eq);

	solver.file_name = "s_h_eq_j_out";
	t1 = omp_get_wtime();
	solver.solve_sequential(/*helmholts_eq=*/eq, /*method =*/"Jacobi");
	t2 = omp_get_wtime();
	solver.output_eq(eq);

	output_file << std::setw(30) << "Sequential LU_dec time: " << std::setw(9)
				<< t2 - t1 << "\n";


	solver.file_name = "s_h_eq_z_out";
	t1 = omp_get_wtime();
	solver.solve_sequential(/*helmholts_eq=*/eq, /*method =*/"Zeidel");
	t2 = omp_get_wtime();

	solver.output_eq(eq);

	output_file << std::setw(30) << "Sequential LU_dec time: " << std::setw(9)
				<< t2 - t1 << "\n";


	solver.file_name = "s_h_eq_zrb_out";
	t1 = omp_get_wtime();
	solver.solve_sequential(/*helmholts_eq=*/eq, /*method =*/"Zeidel_RB");
	t2 = omp_get_wtime();

	solver.output_eq(eq);

	output_file << std::setw(30) << "Sequential LU_dec time: " << std::setw(9)
				<< t2 - t1 << "\n";

	output_file << "\nParallel part: \n";
	// std::vector<std::size_t> threads_vec = {1, 2, 4, 6};  //, 8, 12, 16, 18};
	//
	// for (auto num_threads = threads_vec.begin();
	// 	 num_threads != threads_vec.end(); ++num_threads) {
	// 	std::cout << "\nStart:\tnum_threads = " << *num_threads << "\n";
	// 	omp_set_num_threads(*num_threads);
	//
	// 	output_file << "num_threads = " << *num_threads << "\n";
	// 	t1 = omp_get_wtime();
	// 	t2 = omp_get_wtime();
	// 	output_file << std::setw(30) << "Parallel LU_dec time: " << std::setw(9)
	// 				<< t2 - t1 << "\n";
	// 	std::cout << "\nFinish:\tnum_threads = " << *num_threads << "\n";
	// }

	output_file.close();
	return 0;
}
