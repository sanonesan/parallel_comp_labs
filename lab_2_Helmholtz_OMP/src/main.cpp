#include <omp.h>

#include <cstddef>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "./config.hpp"
#include "./include/Helmholtz/class_Helmholtz.hpp"
#include "./include/Helmholtz/class_Solver_Helmholts.hpp"
#include "./include/Helmholtz/helpers/norm_difference_parallel.hpp"

int main(int argc, char* argv[]) {
	typedef double T;

	double t1, t2;
	std::string algorithm_type, method_type;
	bool OUTPUT_SOLUTION = false;


	Helmholtz_equation<T> eq;
	eq.update_h(0.001);
	eq.update_k(1000.0);


	std::ofstream output_file;
	output_file.open("../output/output_h_001.csv");
	// std::ofstream output_file(config_lab_2_OMP::PATH_output_folder +
	// "/output_h_01.csv");

	// Init csv header
	output_file << "h_lenght,eps,algorithm_type,num_threads,method_type,n_of_"
				   "iter,error_in_"
				   "norm_inf,time_taken\n";

	/*
	 * Output naming format:
	 * A_h_eq_B_out
	 * A = s (Sequential) / p (Parallel)
	 * h_eq = helmholts_equation
	 * B = j (Jacobi) / z (Seidel) / zrb (Zeidel Red-Black iterationd)
	 */
	Solver_Helmoltz_eq<T> solver;
	solver.tol = 1e-6;
	solver.output_folder = "../output/solution";
	// solver.output_folder = config_lab_2_OMP::PATH_output_folder +
	// "/solution";

	solver.notifications = false;


	/**
	 *
	 * Sequential algorithms section
	 *
	 */
	algorithm_type = "Sequential";

	/*
	 * Jacobi
	 */
	solver.file_name = "s_h_eq_j_out";
	solver.compute(eq);
	method_type = "Jacobi";

	t1 = omp_get_wtime();
	solver.solve_sequential(/*helmholts_eq=*/eq, /*method =*/method_type);
	t2 = omp_get_wtime();

	if (OUTPUT_SOLUTION) solver.output_eq(eq);

	output_file << eq._h << "," << solver.tol << "," << algorithm_type << ","
				<< 1 << "," << method_type << "," << solver.iter << ","
				<< norm_difference_parallel(eq._solution_vec, eq.res,
											eq.x1.size(), eq.x2.size())
				<< "," << t2 - t1 << "\n";


	// /*
	//  * Seidel
	//  */
	// solver.file_name = "s_h_eq_z_out";
	// solver.compute(eq);
	// method_type = "Seidel";
	//
	// t1 = omp_get_wtime();
	// solver.solve_sequential(/*helmholts_eq=*/eq, /*method =*/method_type);
	// t2 = omp_get_wtime();
	//
	// if (OUTPUT_SOLUTION) solver.output_eq(eq);
	//
	// output_file << eq._h << "," << solver.tol << "," << algorithm_type << ","
	// 			<< 1 << "," << method_type << "," << solver.iter << ","
	// 			<< norm_difference_parallel(eq._solution_vec, eq.res,
	// 										eq.x1.size(), eq.x2.size())
	// 			<< "," << t2 - t1 << "\n";
	//

	/*
	 * Seidel_RB
	 */
	solver.file_name = "s_h_eq_zrb_out";
	solver.compute(eq);
	method_type = "Seidel_RB";

	t1 = omp_get_wtime();
	solver.solve_sequential(/*helmholts_eq=*/eq, /*method =*/method_type);
	t2 = omp_get_wtime();

	if (OUTPUT_SOLUTION) solver.output_eq(eq);

	output_file << eq._h << "," << solver.tol << "," << algorithm_type << ","
				<< 1 << "," << method_type << "," << solver.iter << ","
				<< norm_difference_parallel(eq._solution_vec, eq.res,
											eq.x1.size(), eq.x2.size())
				<< "," << t2 - t1 << "\n";


	/**
	 *
	 * Parallel algorithms section
	 *
	 */

	algorithm_type = "Parallel";
	std::vector<std::size_t> threads_vec = {1, 2, 4, 6};  //, 8, 12, 16, 18};

	for (auto num_threads = threads_vec.begin();
		 num_threads != threads_vec.end(); ++num_threads) {
		std::cout << "\nStart:\tnum_threads = " << *num_threads << "\n";
		omp_set_num_threads(*num_threads);

		/*
		 * Jacobi
		 */
		solver.file_name = "p_h_eq_j_out";
		solver.compute(eq);
		method_type = "Jacobi";

		t1 = omp_get_wtime();
		solver.solve_parallel(/*helmholts_eq=*/eq, /*method =*/method_type);
		t2 = omp_get_wtime();

		if (OUTPUT_SOLUTION) solver.output_eq(eq);

		output_file << eq._h << "," << solver.tol << "," << algorithm_type
					<< "," << *num_threads << "," << method_type << ","
					<< solver.iter << ","
					<< norm_difference_parallel(eq._solution_vec, eq.res,
												eq.x1.size(), eq.x2.size())
					<< "," << t2 - t1 << "\n";


		// /*
		//  * Seidel
		//  */
		// solver.file_name = "p_h_eq_z_out";
		// solver.compute(eq);
		// method_type = "Seidel";
		//
		// t1 = omp_get_wtime();
		// solver.solve_parallel(/*helmholts_eq=*/eq, /*method =*/method_type);
		// t2 = omp_get_wtime();
		//
		// if (OUTPUT_SOLUTION) solver.output_eq(eq);
		//
		// output_file << eq._h << "," << solver.tol << "," << algorithm_type
		// 			<< "," << *num_threads << "," << method_type << ","
		// 			<< solver.iter << ","
		// 			<< norm_difference_parallel(eq._solution_vec, eq.res,
		// 										eq.x1.size(), eq.x2.size())
		// 			<< "," << t2 - t1 << "\n";

		/*
		 * Seidel_RB
		 */
		solver.file_name = "p_h_eq_zrb_out";
		solver.compute(eq);
		method_type = "Seidel_RB";

		t1 = omp_get_wtime();
		solver.solve_parallel(/*helmholts_eq=*/eq, /*method =*/method_type);
		t2 = omp_get_wtime();

		if (OUTPUT_SOLUTION) solver.output_eq(eq);


		output_file << eq._h << "," << solver.tol << "," << algorithm_type
					<< "," << *num_threads << "," << method_type << ","
					<< solver.iter << ","
					<< norm_difference_parallel(eq._solution_vec, eq.res,
												eq.x1.size(), eq.x2.size())
					<< "," << t2 - t1 << "\n";

		std::cout << "\nFinish:\tnum_threads = " << *num_threads << "\n";
	}

	output_file.close();
	return 0;
}
