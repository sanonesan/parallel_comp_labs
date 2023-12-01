#pragma once
#include <cstdlib>
#include <iostream>
#include <string>

#include "./class_Helmholtz.hpp"
#include "./schemes/Helmholtz_Jacobi_MPI.hpp"
#include "./schemes/Helmholtz_Seidel_MPI.hpp"
#include "./schemes/Helmholtz_sequential.hpp"

template <typename T>
void solve_Helmoltz_eq_parallel_MPI(
	int id, int num_procs, Helmholtz_equation<T>& eq, std::size_t& iter,
	const std::string method = "Seidel_RB",
	const std::string communication_type = "Sendrecv", const T& eps = 1e-6) {
	if (method == "Jacobi") {
		if (communication_type == "Send_Recv") {
			Jacobi_MPI_Send_Recv(id, num_procs, eq, iter, eps);
		} else if (communication_type == "Sendrecv") {
			Jacobi_MPI_Sendrecv(id, num_procs, eq, iter, eps);
		} else if (communication_type == "ISend_IRecv") {
			Jacobi_MPI_ISend_IRecv(id, num_procs, eq, iter, eps);
		}
		return;

	} else if (method == "Seidel_RB") {
		if (communication_type == "Send_Recv") {
			Seidel_RB_MPI_Send_Recv(id, num_procs, eq, iter, eps);
		} else if (communication_type == "Sendrecv") {
			Seidel_RB_MPI_Sendrecv(id, num_procs, eq, iter, eps);
		} else if (communication_type == "ISend_IRecv") {
			Seidel_RB_MPI_ISend_IRecv(id, num_procs, eq, iter, eps);
		}
		return;
	}

	return;
}


template <typename T>
void solve_Helmoltz_eq_sequential(Helmholtz_equation<T>& eq, std::size_t& iter,
								  const std::string method = "Seidel_RB",
								  const T& eps = 1e-6) {
	if (method == "Jacobi") {
		Jacobi(eq, iter, eps);
		return;
	} else if (method == "Seidel") {
		Seidel(eq, iter, eps);
		return;
	} else if (method == "Seidel_RB") {
		Seidel_RB(eq, iter, eps);
		return;
	}

	return;
}
