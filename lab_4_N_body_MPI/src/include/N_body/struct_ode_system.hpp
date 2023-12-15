#pragma once

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

#include "./struct_Body.hpp"
#include "./utils/array_utils.hpp"
#include "./utils/read_write_utils.hpp"


template <typename T>
struct ode_system {
	T tau;
	T t_start, t_final;
	int n_time_steps, N_bodies;
	std::vector<Body<T>> traj_init;

	std::function<void(const std::vector<Body<T>>& traj,
					   std::vector<Body<T>>& diff_traj, std::size_t begin,
					   std::size_t end)>
		N_body_func;

	std::string name;

	ode_system<T> ode_system_default_test(const T time_step = 0.1,
										  const T eps = 1e-6,
										  const T G = 6.67e-11) {
		this->name = "4Body<T>est";
		/**
		 *
		 *
		 *
		 */

		this->t_start = 0.;
		this->t_final = 20.;
		this->tau = time_step;
		this->n_time_steps = (int)((t_final - t_start) / tau);

		std::string path = "./include/N_body/N_body_config/4body.txt";
		this->traj_init = read_data<T>(path);


		auto func = [G, eps](const std::vector<Body<T>>& traj,
							 std::vector<Body<T>>& diff_traj, std::size_t begin,
							 std::size_t end) {
			std::size_t N = traj.size();
			std::array<T, 3> ri_rj, a_tmp;
			T den, c;

			for (std::size_t i = begin; i < end; ++i) {
				diff_traj[i].r = traj[i].v;

				a_tmp.fill(0.);
				for (std::size_t j = 0; j < N; ++j) {
					ri_rj = traj[i].r - traj[j].r;

					den = arr_norm(ri_rj);
					den = std::max(den * den * den, eps * eps * eps);

					ri_rj *= traj[j].m / den;
					a_tmp += ri_rj;
				}
				a_tmp *= -G;
				diff_traj[i].v = a_tmp;
			}
		};

		this->N_body_func = func;

		return *this;
	}


	ode_system<T> ode_system_random_data(const std::size_t N_of_bodies = 1e2,
										 const T time_step = 0.1,
										 const T eps = 1e-6,
										 const T G = 6.67e-11) {
		this->name = "4Body_random";
		/**
		 *
		 *
		 *
		 */

		this->t_start = 0.;
		this->t_final = 20.;
		this->tau = time_step;
		this->n_time_steps = (int)((t_final - t_start) / tau);
		this->N_bodies = N_of_bodies;


		auto func = [G, eps](const std::vector<Body<T>>& traj,
							 std::vector<Body<T>>& diff_traj, std::size_t begin,
							 std::size_t end) {
			std::size_t N = traj.size();
			std::array<T, 3> ri_rj, a_tmp;
			T den, c;

			for (std::size_t i = begin; i < end; ++i) {
				diff_traj[i].r = traj[i].v;

				a_tmp.fill(0.);
				for (std::size_t j = 0; j < N; ++j) {
					ri_rj = traj[i].r - traj[j].r;

					den = arr_norm(ri_rj);
					den = std::max(den * den * den, eps * eps * eps);

					ri_rj *= traj[j].m / den;
					a_tmp += ri_rj;
				}
				a_tmp *= -G;
				diff_traj[i].v = a_tmp;
			}
		};

		this->N_body_func = func;

		return *this;
	}
};
