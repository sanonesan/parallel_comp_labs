#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "omp.h"
using namespace std;

const double Pi = 3.1415926535;
const int k = 2e3;
const double h = 1.0 / k;
const double h2 = h * h;
const double koef = 4.0 + h2 * k * k;
const int n = k + 1;
const double eps = 1e-6;

ostream& operator<<(ostream& out, const vector<double> A) {
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < n; ++i) out << setw(10) << A[j * n + i];
		out << endl;
	}
	out << endl;
	return out;
}

double DifNorm(const vector<double>& A, const vector<double>& B) {
	double norm = 0.0;
	double dd = 0.0;
	int size = A.size();
#pragma omp parallel for default(none) shared(A, B, size) private(dd) \
	reduction(+ : norm)
	for (int i = 0; i < size; ++i) {
		dd = A[i] - B[i];
		norm += dd * dd;
	}
	return sqrt(norm);
}

double f(const double x, const double y) {
	return 2.0 * sin(Pi * y) + k * k * (1.0 - x) * x * sin(Pi * y) +
		   Pi * Pi * (1.0 - x) * sin(Pi * y);
}

void InitConds(vector<double>& u) {
	for (int i = 0; i < n; ++i) {
		u[i] = 0.0;
		u[n * (n - 1) + i] = 0.0;
	}

	for (int j = 1; j < n - 1; ++j) {
		u[j * n] = 0.0;
		u[j * n + n - 1] = 0.0;
	}
}

vector<double> Jacobi(vector<double>& u0) {
	vector<double> u(u0);
	double dif = 0.0;
	int iter = 0;
	do {
		swap(u, u0);

#pragma omp parallel for default(none) shared(u, u0, n, h, h2, koef)
		for (int j = 1; j < n - 1; ++j)
			for (int i = 1; i < n - 1; ++i)
				u[j * n + i] = (h2 * f(h * i, h * j) + u0[j * n + i - 1] +
								u0[j * n + i + 1] + u0[(j - 1) * n + i] +
								u0[(j + 1) * n + i]) /
							   koef;

		// dif = DifNorm(u0, u);
		++iter;
	} while (iter < 50);  //(dif > eps);
	std::cout << "Discrepancy=" << dif << endl;
	std::cout << "N of iterations=" << iter << endl;
	return u;
}

vector<double> Zeidel(vector<double>& u0) {
	vector<double> u(u0);
	double dif = 0.0;
	int iter = 0;
	do {
		swap(u, u0);

		for (int j = 1; j < n - 1; ++j)
			for (int i = 1; i < n - 1; ++i)
				u[j * n + i] = (h2 * f(h * i, h * j) + u[j * n + i - 1] +
								u0[j * n + i + 1] + u[(j - 1) * n + i] +
								u0[(j + 1) * n + i]) /
							   koef;

		dif = DifNorm(u, u0);
		++iter;
	} while (dif > eps);
	std::cout << "Discrepancy=" << dif << endl;
	std::cout << "N of iterations=" << iter << endl;
	return u;
}

vector<double> ZeidelRB(vector<double>& u0) {
	vector<double> u(u0);
	double dif = 0.0;
	int iter = 0;
	do {
		std::swap(u, u0);
		// Обход по красным
#pragma omp parallel for default(none) shared(u, u0, h, h2, n, koef)
		for (int j = 1; j < n - 1; ++j)
			for (int i = 1 + (j - 1) % 2; i < n - 1; i += 2)
				u[j * n + i] = (h2 * f(h * i, h * j) + u0[j * n + i - 1] +
								u0[j * n + i + 1] + u0[(j - 1) * n + i] +
								u0[(j + 1) * n + i]) /
							   koef;

				// Обход по черным
#pragma omp parallel for default(none) shared(u, h, h2, n, koef)
		for (int j = 1; j < n - 1; ++j)
			for (int i = 1 + j % 2; i < n - 1; i += 2)
				u[j * n + i] = (h2 * f(h * i, h * j) + u[j * n + i - 1] +
								u[j * n + i + 1] + u[(j - 1) * n + i] +
								u[(j + 1) * n + i]) /
							   koef;

		// dif = DifNorm(u, u0);
		++iter;
	} while (iter < 27);  //(dif > eps);

	std::cout << "Discrepancy=" << dif << endl;
	std::cout << "N of iterations=" << iter << endl;
	return u;
}
