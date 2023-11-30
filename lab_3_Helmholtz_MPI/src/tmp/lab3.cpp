#include <iostream>

#include "mpi.h"
#define _USE_MATH_DEFINES
#include <math.h>

#include <algorithm>
#include <iomanip>
#include <set>
#include <vector>

using namespace std;

const int N = 3200;
const double h = 1.0 / N;
const double k = 6400;

double u(double x, double y) { return (1 - x) * x * sin(M_PI * y); }

double f(double x, double y) {
	return 2 * sin(M_PI * y) + k * k * (1 - x) * x * sin(M_PI * y) +
		   M_PI * M_PI * (1 - x) * x * sin(M_PI * y);
}

double norm(double* vec, int size) {
	double sum = 0;
	for (size_t i = 0; i < size; i++) {
		sum += vec[i] * vec[i];
	}
	return sqrt(sum);
}

void jacobi(int sendrecvtype)  // 1 - MPI_Send, MPI_Recv;  2 - MPI_Sendrecv;  3
							   // - MPI_Send_init, MPI_Recv_init
{
	int m = N + 1;
	int np, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int nLoc;
	vector<double> sol, solnext, solGlob, deltaGlob, rhs, rhsGlob, exactsol;
	vector<double> points(m);
	vector<int> nLocVec, displs, nLocVecm;
	if (rank == 0) {
		solGlob.resize(m * m);
		deltaGlob.resize(m * m);
		rhsGlob.resize(m * m);
		exactsol.resize(m * m);
		nLocVec.resize(np, m / np);
		for (int i = 0; i < m % np; ++i)  // считаем размер
			nLocVec[i]++;

		for (size_t i = 0; i < m; i++)	// собираем правую часть и точное
										// решение
			points[i] = i * h;
		for (size_t i = 0; i < m; i++)
			for (size_t j = 0; j < m; j++) {
				rhsGlob[i * m + j] = h * h * f(points[i], points[j]);
				exactsol[i * m + j] = u(points[i], points[j]);
			}

		displs.resize(np, 0);  // считаем смещения
		for (int i = 1; i < np; ++i)
			displs[i] = (displs[i - 1] / m + nLocVec[i - 1]) * m;
		nLocVecm.resize(np);
		for (int i = 0; i < nLocVecm.size(); i++) nLocVecm[i] = nLocVec[i] * m;
	}
	MPI_Scatter(nLocVec.data(), 1, MPI_INT, &nLoc, 1, MPI_INT, 0,
				MPI_COMM_WORLD);
	nLoc += 2;
	sol.resize(nLoc * m, 0.0);
	solnext.resize(nLoc * m, 0.0);
	rhs.resize(nLoc * m, 0.0);
	MPI_Scatterv(rhsGlob.data(), nLocVecm.data(), displs.data(), MPI_DOUBLE,
				 rhs.data() + m, (nLoc - 2) * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	set<double> boundaryindex;	// составляем множество индексов граничных узлов
	if (rank == 0) {
		for (int i = 0; i < m; i++)
			boundaryindex.insert(m + i);  /// нижняя граница
		for (int i = 1; i < nLoc;
			 i++)  // m/np строк у каждого + 1 для учета границы верхнего
		{
			boundaryindex.insert(i * m);		  /// левая граница
			boundaryindex.insert(i * m + m - 1);  /// правая граница
		}
		if (np == 1)  // если 1 поток, верхняя граница
		{
			for (int i = 0; i < m; i++)
				boundaryindex.insert((nLoc - 2) * m + i);
		}
	} else if (rank == np - 1) {
		for (int i = 0; i < m; i++)
			boundaryindex.insert((nLoc - 2) * m + i);  /// верхняя граница
		for (int i = 0; i < nLoc - 1; i++) {
			boundaryindex.insert(i * m);		  /// левая граница
			boundaryindex.insert(i * m + m - 1);  /// правая граница
		}
	} else
		for (int i = 0; i < nLoc; i++) {
			boundaryindex.insert(i * m);		  /// левая граница
			boundaryindex.insert(i * m + m - 1);  /// правая граница
		}

	int nL, nR, neibL,
		neibR;	// назначаям "соседей" и количество отправляемых элементов
	nL = (rank == 0) ? 0 : m;
	nR = (rank == np - 1) ? 0 : m;
	neibL = (rank == 0) ? np - 1 : rank - 1;
	neibR = (rank == np - 1) ? 0 : rank + 1;

	MPI_Status st;
	double denominator = 4 + h * h * k * k;
	double N = m * m;
	vector<double> delta(N);  // delta, и flag для критерия останова
	double flag = 1;
	int j = 0;

	double t1 = MPI_Wtime();
	MPI_Status* stSend =
		new MPI_Status[2];	// инициализация  некоторых перменных для пересылки
	MPI_Status* stRecv = new MPI_Status[2];
	MPI_Request *reqSend1, *reqRecv1;
	MPI_Request *reqSend2, *reqRecv2;
	reqSend1 = new MPI_Request[2];
	reqRecv1 = new MPI_Request[2];
	reqSend2 = new MPI_Request[2];
	reqRecv2 = new MPI_Request[2];
	if (np > 1 && sendrecvtype != 1 && sendrecvtype != 2) {
		MPI_Send_init(sol.data() + (nLoc - 2) * m, nR, MPI_DOUBLE, neibR, 42,
					  MPI_COMM_WORLD, reqSend1);
		MPI_Recv_init(sol.data(), nL, MPI_DOUBLE, neibL, 42, MPI_COMM_WORLD,
					  reqRecv1);
		MPI_Send_init(sol.data() + m, nL, MPI_DOUBLE, neibL, 1337,
					  MPI_COMM_WORLD, reqSend1 + 1);
		MPI_Recv_init(sol.data() + (nLoc - 1) * m, nR, MPI_DOUBLE, neibR, 1337,
					  MPI_COMM_WORLD, reqRecv1 + 1);

		MPI_Send_init(solnext.data() + (nLoc - 2) * m, nR, MPI_DOUBLE, neibR,
					  42, MPI_COMM_WORLD, reqSend2);
		MPI_Recv_init(solnext.data(), nL, MPI_DOUBLE, neibL, 42, MPI_COMM_WORLD,
					  reqRecv2);
		MPI_Send_init(solnext.data() + m, nL, MPI_DOUBLE, neibL, 1337,
					  MPI_COMM_WORLD, reqSend2 + 1);
		MPI_Recv_init(solnext.data() + (nLoc - 1) * m, nR, MPI_DOUBLE, neibR,
					  1337, MPI_COMM_WORLD, reqRecv2 + 1);
	}
	do {
		++j;
		if (np > 1) {
			if (sendrecvtype == 1)	// 1 - MPI_Send, MPI_Recv;
			{
				MPI_Send(sol.data() + (nLoc - 2) * m, nR, MPI_DOUBLE, neibR, 42,
						 MPI_COMM_WORLD);
				MPI_Recv(sol.data(), nL, MPI_DOUBLE, neibL, 42, MPI_COMM_WORLD,
						 &st);
				MPI_Send(sol.data() + m, nL, MPI_DOUBLE, neibL, 1337,
						 MPI_COMM_WORLD);
				MPI_Recv(sol.data() + (nLoc - 1) * m, nR, MPI_DOUBLE, neibR,
						 1337, MPI_COMM_WORLD, &st);
			} else if (sendrecvtype == 2)  // 2 - MPI_Sendrecv;
			{
				MPI_Sendrecv(sol.data() + (nLoc - 2) * m, nR, MPI_DOUBLE, neibR,
							 42, sol.data(), nL, MPI_DOUBLE, neibL, 42,
							 MPI_COMM_WORLD, &st);
				MPI_Sendrecv(sol.data() + m, nL, MPI_DOUBLE, neibL, 1337,
							 sol.data() + (nLoc - 1) * m, nR, MPI_DOUBLE, neibR,
							 1337, MPI_COMM_WORLD, &st);
			} else	// 3 - MPI_Send_init, MPI_Recv_init
			{
				if (j % 2 == 1) {
					MPI_Startall(2, reqSend1);
					MPI_Startall(2, reqRecv1);
					//	MPI_Waitall(2, reqSend1, stSend);
					//	MPI_Waitall(2, reqRecv1, stSend);
				} else {
					MPI_Startall(2, reqSend2);
					MPI_Startall(2, reqRecv2);
					//	MPI_Waitall(2, reqSend2, stSend);
					//	MPI_Waitall(2, reqRecv2, stSend);
				}
			}
		}

		for (int i = 2 * m; i < (nLoc - 2) * m;
			 ++i)  // решаем внутри области, без граничных строк
			if (boundaryindex.count(i) == 0)
				solnext[i] = (rhs[i] + sol[i - 1] + sol[i + 1] + sol[i + m] +
							  sol[i - m]) /
							 denominator;


		if (np > 1 && sendrecvtype == 3) {
			if (j % 2 == 1) {
				MPI_Waitall(2, reqSend1, stSend);
				MPI_Waitall(2, reqRecv1, stSend);
			} else {
				MPI_Waitall(2, reqSend2, stSend);
				MPI_Waitall(2, reqRecv2, stSend);
			}
		}

		for (int i = m; i < 2 * m; ++i)	 // решаем на нижней строке
			if (boundaryindex.count(i) == 0)
				solnext[i] = (rhs[i] + sol[i - 1] + sol[i + 1] + sol[i + m] +
							  sol[i - m]) /
							 denominator;
		for (int i = (nLoc - 2) * m; i < (nLoc - 1) * m;
			 ++i)  // решаем на верхней строке
			if (boundaryindex.count(i) == 0)
				solnext[i] = (rhs[i] + sol[i - 1] + sol[i + 1] + sol[i + m] +
							  sol[i - m]) /
							 denominator;
		sol.swap(solnext);

		double a1, a2, b1, b2;
		if (np > 1) {
			// вычисляем локально нормы решения и нормы ошибки
			for (int i = 0; i < m * nLoc; ++i) delta[i] = solnext[i] - sol[i];
			a1 = norm(delta.data(), delta.size());
			a2 = norm(sol.data(), sol.size());
			// суммируем локальные нормы ошибки и решения
			MPI_Reduce(&a1, &b1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&a2, &b2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

			if (rank == 0) {
				// cout << rank << ": b1: " << b1 << ",  b2: " << b2 << "    "
				// << b1/b2 << endl;
				//	cout << rank << ": a1: " << a1 << ",  a2: " << a2 << "    "
				//<< a1 / a2 << endl;
			}
		} else {
			for (int i = 0; i < m * m;
				 ++i)  // для одного узла собираем решение (np=1)
			{
				solGlob[i] = sol[m + i];
				deltaGlob[i] = solnext[m + i] - sol[m + i];
			}
			b1 = norm(deltaGlob.data(), deltaGlob.size());
			b2 = norm(solGlob.data(), solGlob.size());
			if (rank == 0) {
				// cout << j << "   " << rank << ": b1: " << b1 << ",  b2: " <<
				// b2 << "    " << b1 / b2 << endl;
			}
		}

		if (rank == 0)	// оцениваем ошибку соседних итераций
		{
			if (b1 / b2 < 1E-6) flag = 0;
		}
		if (np > 1)
			MPI_Bcast(&flag, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);	 // сообщаем
																 // всем
	} while (flag);

	if (np > 1)
		MPI_Gatherv(sol.data() + m, (nLoc - 2) * m, MPI_DOUBLE, solGlob.data(),
					nLocVecm.data(), displs.data(), MPI_DOUBLE, 0,
					MPI_COMM_WORLD);

	double t2 = MPI_Wtime();
	if (rank == 0) {
		if (sendrecvtype == 1)	// 1 - MPI_Send, MPI_Recv;
			cout << "Jacobi MPI_Send, MPI_Recv: " << endl;
		else if (sendrecvtype == 2)	 // 2 - MPI_Sendrecv;
			cout << "Jacobi MPI_Sendrecv: " << endl;
		else  // 3 - MPI_Send_init, MPI_Recv_init
			cout << "Jacobi MPI_Send_init, MPI_Recv_init: " << endl;

		cout << "Iter = " << j << endl;
		cout << "Time = " << t2 - t1 << endl;
		vector<double> delta(m * m);
		for (int i = 0; i < m * m; ++i)
			delta[i] = fabs(solGlob[i] - exactsol[i]);
		cout << "Error = "
			 << *max_element(delta.begin(), delta.end()) /
					norm(exactsol.data(), N)
			 << endl
			 << endl;
	}
}

void zeidel(int sendrecvtype)  // 1 - MPI_Send, MPI_Recv;  2 - MPI_Sendrecv;  3
							   // - MPI_Send_init, MPI_Recv_init
{
	int m = N + 1;
	int np, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int nLoc;
	vector<double> sol, solnext, solGlob, deltaGlob, rhs, rhsGlob, exactsol;
	vector<double> points(m);
	vector<int> nLocVec, displs, nLocVecm;
	if (rank == 0) {
		solGlob.resize(m * m);
		deltaGlob.resize(m * m);
		rhsGlob.resize(m * m);
		exactsol.resize(m * m);
		nLocVec.resize(np, m / np);
		for (int i = 0; i < m % np; ++i)  // считаем размер
			nLocVec[i]++;

		for (size_t i = 0; i < m; i++)	// собираем правую часть и точное
										// решение
			points[i] = i * h;
		for (size_t i = 0; i < m; i++)
			for (size_t j = 0; j < m; j++) {
				rhsGlob[i * m + j] = h * h * f(points[i], points[j]);
				exactsol[i * m + j] = u(points[i], points[j]);
			}

		displs.resize(np, 0);  // считаем смещения
		for (int i = 1; i < np; ++i)
			displs[i] = (displs[i - 1] / m + nLocVec[i - 1]) * m;
		nLocVecm.resize(np);
		for (int i = 0; i < nLocVecm.size(); i++) nLocVecm[i] = nLocVec[i] * m;
	}
	MPI_Scatter(nLocVec.data(), 1, MPI_INT, &nLoc, 1, MPI_INT, 0,
				MPI_COMM_WORLD);
	nLoc += 2;
	sol.resize(nLoc * m, 0.0);
	solnext.resize(nLoc * m, 0.0);
	rhs.resize(nLoc * m, 0.0);
	MPI_Scatterv(rhsGlob.data(), nLocVecm.data(), displs.data(), MPI_DOUBLE,
				 rhs.data() + m, (nLoc - 2) * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	set<double> boundaryindex;	// составляем множество индексов граничных узлов
	if (rank == 0) {
		for (int i = 0; i < m; i++)
			boundaryindex.insert(m + i);  /// нижняя граница
		for (int i = 1; i < nLoc;
			 i++)  // m/np строк у каждого + 1 для учета границы верхнего
		{
			boundaryindex.insert(i * m);		  /// левая граница
			boundaryindex.insert(i * m + m - 1);  /// правая граница
		}
		if (np == 1)  // если 1 поток, верхняя граница
		{
			for (int i = 0; i < m; i++)
				boundaryindex.insert((nLoc - 2) * m + i);
		}
	} else if (rank == np - 1) {
		for (int i = 0; i < m; i++)
			boundaryindex.insert((nLoc - 2) * m + i);  /// верхняя граница
		for (int i = 0; i < nLoc - 1; i++) {
			boundaryindex.insert(i * m);		  /// левая граница
			boundaryindex.insert(i * m + m - 1);  /// правая граница
		}
	} else
		for (int i = 0; i < nLoc; i++) {
			boundaryindex.insert(i * m);		  /// левая граница
			boundaryindex.insert(i * m + m - 1);  /// правая граница
		}

	int nL, nR, neibL,
		neibR;	// назначаям "соседей" и количество отправляемых элементов
	nL = (rank == 0) ? 0 : m;
	nR = (rank == np - 1) ? 0 : m;
	neibL = (rank == 0) ? np - 1 : rank - 1;
	neibR = (rank == np - 1) ? 0 : rank + 1;

	MPI_Status st;
	double denominator = 4 + h * h * k * k;
	double N = m * m;
	vector<double> delta(nLoc * m);	 // delta, и flag для критерия останова
	double flag = 1;
	int j = 0;

	double t1 = MPI_Wtime();
	MPI_Status* stSend =
		new MPI_Status[2];	// инициализация  некоторых перменных для пересылки
	MPI_Status* stRecv = new MPI_Status[2];
	MPI_Request *reqSend1, *reqRecv1;
	MPI_Request *reqSend2, *reqRecv2;
	reqSend1 = new MPI_Request[2];
	reqRecv1 = new MPI_Request[2];
	reqSend2 = new MPI_Request[2];
	reqRecv2 = new MPI_Request[2];
	if (np > 1 && sendrecvtype != 1 && sendrecvtype != 2) {
		MPI_Send_init(sol.data() + (nLoc - 2) * m, nR, MPI_DOUBLE, neibR, 42,
					  MPI_COMM_WORLD, reqSend1);
		MPI_Recv_init(sol.data(), nL, MPI_DOUBLE, neibL, 42, MPI_COMM_WORLD,
					  reqRecv1);
		MPI_Send_init(sol.data() + m, nL, MPI_DOUBLE, neibL, 1337,
					  MPI_COMM_WORLD, reqSend1 + 1);
		MPI_Recv_init(sol.data() + (nLoc - 1) * m, nR, MPI_DOUBLE, neibR, 1337,
					  MPI_COMM_WORLD, reqRecv1 + 1);

		MPI_Send_init(solnext.data() + (nLoc - 2) * m, nR, MPI_DOUBLE, neibR,
					  42, MPI_COMM_WORLD, reqSend2);
		MPI_Recv_init(solnext.data(), nL, MPI_DOUBLE, neibL, 42, MPI_COMM_WORLD,
					  reqRecv2);
		MPI_Send_init(solnext.data() + m, nL, MPI_DOUBLE, neibL, 1337,
					  MPI_COMM_WORLD, reqSend2 + 1);
		MPI_Recv_init(solnext.data() + (nLoc - 1) * m, nR, MPI_DOUBLE, neibR,
					  1337, MPI_COMM_WORLD, reqRecv2 + 1);
	}
	do {
		++j;
		if (np > 1) {
			if (sendrecvtype == 1)	// 1 - MPI_Send, MPI_Recv;
			{
				MPI_Send(sol.data() + (nLoc - 2) * m, nR, MPI_DOUBLE, neibR, 42,
						 MPI_COMM_WORLD);
				MPI_Recv(sol.data(), nL, MPI_DOUBLE, neibL, 42, MPI_COMM_WORLD,
						 &st);
				MPI_Send(sol.data() + m, nL, MPI_DOUBLE, neibL, 1337,
						 MPI_COMM_WORLD);
				MPI_Recv(sol.data() + (nLoc - 1) * m, nR, MPI_DOUBLE, neibR,
						 1337, MPI_COMM_WORLD, &st);
			} else if (sendrecvtype == 2)  // 2 - MPI_Sendrecv;
			{
				MPI_Sendrecv(sol.data() + (nLoc - 2) * m, nR, MPI_DOUBLE, neibR,
							 42, sol.data(), nL, MPI_DOUBLE, neibL, 42,
							 MPI_COMM_WORLD, &st);
				MPI_Sendrecv(sol.data() + m, nL, MPI_DOUBLE, neibL, 1337,
							 sol.data() + (nLoc - 1) * m, nR, MPI_DOUBLE, neibR,
							 1337, MPI_COMM_WORLD, &st);
			} else	// 3 - MPI_Send_init, MPI_Recv_init
			{
				if (j % 2 == 1) {
					MPI_Startall(2, reqSend1);
					MPI_Startall(2, reqRecv1);
					// MPI_Waitall(2, reqSend1, stSend);
					// MPI_Waitall(2, reqRecv1, stSend);
				} else {
					MPI_Startall(2, reqSend2);
					MPI_Startall(2, reqRecv2);
					// MPI_Waitall(2, reqSend2, stSend);
					// MPI_Waitall(2, reqRecv2, stSend);
				}
			}
		}

		for (int i = m; i < (nLoc - 1) * m; i += 2)
			if (boundaryindex.count(i) == 0)
				solnext[i] = (rhs[i] + sol[i - 1] + sol[i + 1] + sol[i + m] +
							  sol[i - m]) /
							 denominator;

		if (np > 1) {
			if (sendrecvtype == 1)	// 1 - MPI_Send, MPI_Recv;
			{
				MPI_Send(solnext.data() + (nLoc - 2) * m, nR, MPI_DOUBLE, neibR,
						 42, MPI_COMM_WORLD);
				MPI_Recv(solnext.data(), nL, MPI_DOUBLE, neibL, 42,
						 MPI_COMM_WORLD, &st);
				MPI_Send(solnext.data() + m, nL, MPI_DOUBLE, neibL, 1337,
						 MPI_COMM_WORLD);
				MPI_Recv(solnext.data() + (nLoc - 1) * m, nR, MPI_DOUBLE, neibR,
						 1337, MPI_COMM_WORLD, &st);
			} else if (sendrecvtype == 2)  // 2 - MPI_Sendrecv;
			{
				MPI_Sendrecv(solnext.data() + (nLoc - 2) * m, nR, MPI_DOUBLE,
							 neibR, 42, solnext.data(), nL, MPI_DOUBLE, neibL,
							 42, MPI_COMM_WORLD, &st);
				MPI_Sendrecv(solnext.data() + m, nL, MPI_DOUBLE, neibL, 1337,
							 solnext.data() + (nLoc - 1) * m, nR, MPI_DOUBLE,
							 neibR, 1337, MPI_COMM_WORLD, &st);
			} else	// 3 - MPI_Send_init, MPI_Recv_init
			{
				if (j % 2 == 1) {
					// MPI_Startall(2, reqSend2);
					// MPI_Startall(2, reqRecv2);
					MPI_Waitall(2, reqSend2, stSend);
					MPI_Waitall(2, reqRecv2, stSend);
				} else {
					// MPI_Startall(2, reqSend1);
					// MPI_Startall(2, reqRecv1);
					MPI_Waitall(2, reqSend1, stSend);
					MPI_Waitall(2, reqRecv1, stSend);
				}
			}
		}

		for (int i = m + 1; i < (nLoc - 1) * m; i += 2)
			if (boundaryindex.count(i) == 0)
				solnext[i] = (rhs[i] + solnext[i - 1] + solnext[i + 1] +
							  solnext[i + m] + solnext[i - m]) /
							 denominator;

		sol.swap(solnext);

		double a1, a2, b1, b2;
		if (np > 1) {
			// вычисляем локально нормы решения и нормы ошибки
			for (int i = 0; i < m * nLoc; ++i) delta[i] = solnext[i] - sol[i];
			a1 = norm(delta.data(), delta.size());
			a2 = norm(sol.data(), sol.size());
			// суммируем локальные нормы ошибки и решения
			MPI_Reduce(&a1, &b1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&a2, &b2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

			//	if (rank == 0)	cout << rank << ": b1: " << b1 << ",  b2: " <<
			// b2 << "    " << b1 / b2 << endl;
		} else {
			for (int i = 0; i < m * m;
				 ++i)  // для одного узла собираем решение (np=1)
			{
				solGlob[i] = sol[m + i];
				deltaGlob[i] = solnext[m + i] - sol[m + i];
			}
			b1 = norm(deltaGlob.data(), deltaGlob.size());
			b2 = norm(solGlob.data(), solGlob.size());
			//			if (rank == 0)	cout << rank << ": b1: " << a1 << ", b2:
			//" << a2 << "    " << b1 / b2 << endl;
		}

		if (rank == 0)	// оцениваем ошибку соседних итераций
		{
			if (b1 / b2 < 1E-6) flag = 0;
		}
		if (np > 1)
			MPI_Bcast(&flag, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);	 // сообщаем
																 // всем
	} while (flag);

	if (np > 1)
		MPI_Gatherv(sol.data() + m, (nLoc - 2) * m, MPI_DOUBLE, solGlob.data(),
					nLocVecm.data(), displs.data(), MPI_DOUBLE, 0,
					MPI_COMM_WORLD);

	double t2 = MPI_Wtime();
	if (rank == 0) {
		if (sendrecvtype == 1)	// 1 - MPI_Send, MPI_Recv;
			cout << "Ziedel MPI_Send, MPI_Recv: " << endl;
		else if (sendrecvtype == 2)	 // 2 - MPI_Sendrecv;
			cout << "Ziedel MPI_Sendrecv: " << endl;
		else  // 3 - MPI_Send_init, MPI_Recv_init
			cout << "Ziedel MPI_Send_init, MPI_Recv_init: " << endl;

		cout << "Iter = " << j << endl;
		cout << "Time = " << t2 - t1 << endl;
		vector<double> delta(m * m);
		for (int i = 0; i < m * m; ++i)
			delta[i] = fabs(solGlob[i] - exactsol[i]);
		cout << "Error = "
			 << *max_element(delta.begin(), delta.end()) /
					norm(exactsol.data(), N)
			 << endl
			 << endl;
	}
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int np, rank;

	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	jacobi(1);	// 1 - MPI_Send, MPI_Recv;  2 - MPI_Sendrecv;  3 -
				// MPI_Send_init, MPI_Recv_init
	jacobi(2);
	jacobi(3);

	zeidel(1);	// 1 - MPI_Send, MPI_Recv;  2 - MPI_Sendrecv;  3 -
				// MPI_Send_init, MPI_Recv_init
	zeidel(2);
	zeidel(3);


	MPI_Finalize();

	return 0;
}
