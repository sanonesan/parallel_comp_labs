#include <mpi.h>

#include <iostream>
#include <vector>

#include "math.h"

using namespace std;
const double PI = 3.14159265359;
const double eps = 1e-6;
const int n = 10e3;
const double k = n - 1;
const double h = 1.0 / (n - 1);

double hh = h * h;
double q = 1.0 / (4.0 + k * k * hh);

double f(double x, double y) {
	return 2.0 * sin(PI * y) + k * k * (1.0 - x) * x * sin(PI * y) +
		   PI * PI * (1.0 - x) * x * sin(PI * y);
}

double PreciseSol(double x, double y) { return (1.0 - x) * x * sin(PI * y); }

double error(const vector<double>& y0, const vector<double>& y1) {
	double max = 0.0, temp;
	for (int i = 0; i < y0.size(); ++i) {
		temp = fabs(y0[i] - y1[i]);
		if (temp > max) max = temp;
	}
	return max;
}

double norml2(const vector<double>& rez) {
	double sum = 0.0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			sum += (rez[i * n + j] - PreciseSol(i * h, j * h)) *
				   (rez[i * n + j] - PreciseSol(i * h, j * h));
		}
	}
	return sqrt(sum);
}


double norm1(const vector<double>& rez) {
	double max = 0.0;
	double t;

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			t = fabs(rez[i * n + j] - PreciseSol(i * h, j * h));
			if (t > max) max = t;
		}
	}
	return max;
}


void Jacobi1(int myid,
			 int np) {	// np - количество процессоров, myid - номер процессора

	double err = 0.0;
	int iteration = 0;
	vector<double> solution;  // решение
	vector<int> locSize,
		offset;	 // locSize- хранит размеры строк, которые обрабатывает
				 // конкретный процессор  offset-смещение(начиная с какой строки
				 // процессор берет себе строки)

	if (myid == 0) {
		solution.resize(n * n);
		offset.resize(np);
		locSize.resize(np, 0);

		for (int p = 0; p < np - 1; ++p) {
			locSize[p] = n / np;
			offset[p + 1] = offset[p] + locSize[p];
		}
		locSize[np - 1] = n / np + n % np;
	}
	int sizeparts;	// размеры массивов частей решений для каждого из
					// процессоров
	int myOffset;

	MPI_Scatter(
		locSize.data(), 1, MPI_INT, &sizeparts, 1, MPI_INT, 0,
		MPI_COMM_WORLD);  // это рассылает 0 процессор всем процессорам
						  // количество строк которое он должен обработать (у
						  // каждого процессора это в sizeparts)
	vector<double> localsolution(sizeparts * n, 0.0);  // текущее решение
	vector<double> solutionold(sizeparts * n,
							   0.0);  // решение с предыдущей итерации

	MPI_Scatter(offset.data(), 1, MPI_INT, &myOffset, 1, MPI_INT, 0,
				MPI_COMM_WORLD);


	vector<double> lineabove(n),
		linebelow(n);  // строка сверху и снизу для процессора
	double t1 = -MPI_Wtime();

	// решение  для каждого процессора

	double solNorm, diffNorm, solution_norm, difference_norm, err_local;
	/////////вводим переменные для сокращщения и обобщения кода процессор
	/// отправляет верхнюю строчку вверх а нижнюю вниз, т.е. n данных нижнему
	/// процессору
	int source = myid ? myid - 1 : np - 1;
	int dest = (myid != (np - 1)) ? myid + 1 : 0;
	// чтобы обобщить работу 0, последнего и остальных процессоров
	int sendcount = (myid != (np - 1)) ? n : 0;
	int recvcount =
		myid ? n : 0;  // 0 процесс вниз иечего не отправляет, т.е. 0 данных
	// первый процессор отправляет нижнюю строку второу а верхнюю первому
	do {
		++iteration;
		// перессылка данных
		if (np > 1) {
			MPI_Send(solutionold.data() + (sizeparts - 1) * n, sendcount,
					 MPI_DOUBLE, dest, 111, MPI_COMM_WORLD);
			MPI_Recv(lineabove.data(), recvcount, MPI_DOUBLE, source, 111,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
			// сначала указатель на элемент который стоит первым в последней
			// строке , процессор передает следующему n элементов(нижнюю
			// строку), она должна перейти в lineabove
			MPI_Send(solutionold.data(), recvcount, MPI_DOUBLE, source, 222,
					 MPI_COMM_WORLD);
			MPI_Recv(linebelow.data(), sendcount, MPI_DOUBLE, dest, 222,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}

		for (int i = 1; i < sizeparts - 1;
			 ++i) {	 // вычисление во внутренних узлах полосы от 1 до sizeparts
					 // - 2 включительно
			for (int j = 1; j < n - 1; ++j) {
				localsolution[i * n + j] =
					q * (solutionold[(i + 1) * n + j] +
						 solutionold[(i - 1) * n + j] +
						 solutionold[i * n + (j + 1)] +
						 solutionold[i * n + (j - 1)] +
						 hh * f((i + myOffset) * h, j * h));
			}
		}

		if (myid != 0) {  // если процесс ненулевой, то можем производить
						  // вычисление в верхних строчках, т.к. у нулевого нет
						  // верхней дополнительной строки
			for (int j = 1; j < n - 1; ++j) {
				localsolution[0 * n + j] =
					q * (solutionold[n + j] + lineabove[j] +
						 solutionold[0 * n + (j + 1)] +
						 solutionold[0 * n + (j - 1)] +
						 hh * f((0 + myOffset) * h, j * h));
			}
		}
		if (myid != np - 1) {  // у последнего нет нижней дополнительной строки
							   // (она отвечает за ГУ)
			for (int j = 1; j < n - 1; ++j) {
				localsolution[(sizeparts - 1) * n + j] =
					q * (linebelow[j] + solutionold[(sizeparts - 2) * n + j] +
						 solutionold[(sizeparts - 1) * n + (j + 1)] +
						 solutionold[(sizeparts - 1) * n + (j - 1)] +
						 hh * f(((sizeparts - 1) + myOffset) * h, j * h));
			}
		}

		// находим максимальную разность ошибки, присваиваем  err,
		// распространить  среди всех процессоров
		double localErr = error(localsolution, solutionold);
		MPI_Allreduce(&localErr, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		swap(localsolution, solutionold);
	} while (err > eps);
	// t1 += MPI_Wtime();
	t1 += MPI_Wtime();
	// изменяем массив locSize, чтобы он хранил количество элементов для каждого
	// процессора. До этого хранил количество обрабатываемых строк
	// нулевой массив через Gatherv собирает решение. ему нужно понимать по
	// скольку элементам от каждого процессора брать и добавлять в sol и начиная
	// с какого индекса их располагать
	if (myid == 0) {
		for (auto& x : locSize)
			x *= n;	 // узнаем сколько процессор обрабатывает элементов
		for (auto& x : offset) x *= n;
	}
	// собираем решение (Gatherv предоставляет возможность сбора данных, когда
	// размеры передаваемых процессорами сообщений могут быть различны)
	MPI_Gatherv(solutionold.data(), sizeparts * n, MPI_DOUBLE, solution.data(),
				locSize.data(), offset.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (myid == 0) {
		// 1 векторная норма, то есть максимальный элемент вектора составленного
		// из разности точного и приближенного решения
		double norm = norm1(solution);	//
										// double norm = norml2(solution);//
		cout << "---Jacobi MPI_Send&MPI_Recv---" << endl;
		cout << "n of iterations: " << iteration << endl;

		cout << "Discrepancy=" << norm << endl;
		cout << "time=" << t1 << endl;
		cout << endl;
	}
}


void Jacobi2(int myid, int np) {
	double err = 0.0;
	int iteration = 0;
	vector<double> solution;
	vector<int> locSize, offset;

	if (myid == 0) {
		solution.resize(n * n);
		offset.resize(np);
		locSize.resize(np, 0);

		for (int p = 0; p < np - 1; ++p) {
			locSize[p] = n / np;
			offset[p + 1] = offset[p] + locSize[p];
		}
		locSize[np - 1] = n / np + n % np;
	}
	int sizeparts;	// размеры массивов частей решений для каждого из
					// процессоров
	int myOffset;

	MPI_Scatter(locSize.data(), 1, MPI_INT, &sizeparts, 1, MPI_INT, 0,
				MPI_COMM_WORLD);
	vector<double> localsolution(
		sizeparts * n, 0.0);  // локальное решение для каждого из процессоров
	vector<double> solutionold(sizeparts * n, 0.0);

	MPI_Scatter(offset.data(), 1, MPI_INT, &myOffset, 1, MPI_INT, 0,
				MPI_COMM_WORLD);


	vector<double> lineabove(n), linebelow(n);
	double t1 = -MPI_Wtime();
	// тут решение системы для каждого процессора

	double solNorm, diffNorm, solution_norm, difference_norm, err_local;


	//////
	int sendCount = (myid != (np - 1)) ? n : 0, recvCount = myid ? n : 0;
	int dest = (myid != (np - 1)) ? myid + 1 : 0,
		source = myid ? myid - 1 : np - 1;	// кому, откуда
	do {
		++iteration;

		MPI_Sendrecv(solutionold.data() + (sizeparts - 1) * n, sendCount,
					 MPI_DOUBLE, dest, 111, lineabove.data(), recvCount,
					 MPI_DOUBLE, source, 111, MPI_COMM_WORLD,
					 MPI_STATUSES_IGNORE);
		MPI_Sendrecv(solutionold.data(), recvCount, MPI_DOUBLE, source, 222,
					 linebelow.data(), sendCount, MPI_DOUBLE, dest, 222,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);


		for (int i = 1; i < sizeparts - 1; ++i) {
			for (int j = 1; j < n - 1; ++j) {
				localsolution[i * n + j] =
					q * (solutionold[(i + 1) * n + j] +
						 solutionold[(i - 1) * n + j] +
						 solutionold[i * n + (j + 1)] +
						 solutionold[i * n + (j - 1)] +
						 hh * f((i + myOffset) * h, j * h));
			}
		}

		if (myid != 0) {
			for (int j = 1; j < n - 1; ++j) {
				localsolution[0 * n + j] =
					q * (solutionold[n + j] + lineabove[j] +
						 solutionold[0 * n + (j + 1)] +
						 solutionold[0 * n + (j - 1)] +
						 hh * f((0 + myOffset) * h, j * h));
			}
		}
		if (myid != np - 1) {
			for (int j = 1; j < n - 1; ++j) {
				localsolution[(sizeparts - 1) * n + j] =
					q * (linebelow[j] + solutionold[(sizeparts - 2) * n + j] +
						 solutionold[(sizeparts - 1) * n + (j + 1)] +
						 solutionold[(sizeparts - 1) * n + (j - 1)] +
						 hh * f(((sizeparts - 1) + myOffset) * h, j * h));
			}
		}


		double localErr = error(localsolution, solutionold);
		MPI_Allreduce(&localErr, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		swap(localsolution, solutionold);
	} while (err > eps);
	// t1 += MPI_Wtime();


	if (myid == 0) {
		for (auto& x : locSize) x *= n;
		for (auto& x : offset) x *= n;
	}
	// собираем решение
	MPI_Gatherv(solutionold.data(), sizeparts * n, MPI_DOUBLE, solution.data(),
				locSize.data(), offset.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	t1 += MPI_Wtime();
	if (myid == 0) {
		double norm = norm1(solution);
		cout << "---Jacobi MPI_Sendrecv---" << endl;
		cout << "n of iterations: " << iteration << endl;

		cout << "Discrepancy=" << norm << endl;
		cout << "time=" << t1 << endl;
		cout << endl;
	}
}


void Jacobi3(int myid, int np) {
	double err = 0.0;
	int iter = 0;
	vector<double> solution;
	vector<int> locSize, offset;

	if (myid == 0) {
		solution.resize(n * n);
		offset.resize(
			np);  // locSize- хранит размеры строк,которые обрабатывает
				  // конкретный процессор  offset-смещение(начиная с какой
				  // строки процессор берет себе строки)

		locSize.resize(np, 0);

		for (int p = 0; p < np - 1; ++p) {
			locSize[p] = n / np;
			offset[p + 1] = offset[p] + locSize[p];
		}
		locSize[np - 1] = n / np + n % np;
	}
	int sizeparts;	// размеры массивов частей решений для каждого из
					// процессоров
	int myOffset;

	MPI_Scatter(locSize.data(), 1, MPI_INT, &sizeparts, 1, MPI_INT, 0,
				MPI_COMM_WORLD);
	vector<double> localsolution(
		sizeparts * n, 0.0);  // локальное решение для каждого из процессоров
	vector<double> localsolutionOld(sizeparts * n,
									0.0);  // для предыдущей итерации

	MPI_Scatter(
		offset.data(), 1, MPI_INT, &myOffset, 1, MPI_INT, 0,
		MPI_COMM_WORLD);  // это рассылает 0 процессор всем процессорам
						  // количество строк которое он должен обработать (у
						  // каждого процессора это в sizeparts)


	vector<double> lineabove(n),
		linebelow(n);  // строка снизу и сверху для каждого из процессоров
	double t1 = -MPI_Wtime();


	double solNorm, diffNorm, solution_norm, difference_norm, err_local;

	MPI_Request *reqSend1, *reqRecv1, *reqSend2,
		*reqRecv2;	// формируем массив запросов на отправку и получение

	reqSend1 =
		new MPI_Request[1];	 // 2 можно сделать по 2, только в случае отправки
							 // второй строки для крайних процессоров будут
							 // отсылаться нули, что следует из условий ниже
	reqRecv1 = new MPI_Request[1];	// 2
	reqSend2 = new MPI_Request[2];
	reqRecv2 = new MPI_Request[2];

	int idTop, idBtm, topSize, btmSize;

	idTop = (myid != 0) ? myid - 1 : np - 1;  // условия отсылки
	idBtm = (myid != np - 1) ? myid + 1 : 0;

	topSize = (myid != 0) ? n : 0;
	btmSize = (myid != np - 1) ? n : 0;


	/////////////////
	MPI_Send_init(localsolutionOld.data(), topSize, MPI_DOUBLE, idTop, 111,
				  MPI_COMM_WORLD, reqSend1);
	MPI_Recv_init(linebelow.data(), btmSize, MPI_DOUBLE, idBtm, MPI_ANY_TAG,
				  MPI_COMM_WORLD, reqRecv1);

	MPI_Send_init(localsolutionOld.data() + (sizeparts - 1) * n, btmSize,
				  MPI_DOUBLE, idBtm, 111, MPI_COMM_WORLD, reqSend1 + 1);
	MPI_Recv_init(lineabove.data(), topSize, MPI_DOUBLE, idTop, MPI_ANY_TAG,
				  MPI_COMM_WORLD, reqRecv1 + 1);

	MPI_Send_init(localsolution.data(), topSize, MPI_DOUBLE, idTop, 111,
				  MPI_COMM_WORLD, reqSend2);
	MPI_Recv_init(linebelow.data(), btmSize, MPI_DOUBLE, idBtm, MPI_ANY_TAG,
				  MPI_COMM_WORLD, reqRecv2);

	MPI_Send_init(localsolution.data() + (sizeparts - 1) * n, btmSize,
				  MPI_DOUBLE, idBtm, 111, MPI_COMM_WORLD, reqSend2 + 1);
	MPI_Recv_init(lineabove.data(), topSize, MPI_DOUBLE, idTop, MPI_ANY_TAG,
				  MPI_COMM_WORLD, reqRecv2 + 1);
	////////////////

	do {
		++iter;
		swap(localsolution,
			 localsolutionOld);	 // меняем указатели местами то есть новое
								 // решение и сторое меняетчя местами каждые две
								 // итерации, поэтому отправляем либо из массива
								 // localsolution либо из localsolutionOld

		if (iter % 2 == 0) {
			MPI_Startall(2, reqSend1);
			MPI_Startall(2, reqRecv1);
		} else {
			MPI_Startall(2, reqSend2);
			MPI_Startall(2, reqRecv2);
		}


		for (int i = 1; i < sizeparts - 1;
			 ++i) {	 // во внутренних строках полосы
			for (int j = 1; j < n - 1; ++j) {
				localsolution[i * n + j] =
					q * (localsolutionOld[(i + 1) * n + j] +
						 localsolutionOld[(i - 1) * n + j] +
						 localsolutionOld[i * n + (j + 1)] +
						 localsolutionOld[i * n + (j - 1)] +
						 hh * f((i + myOffset) * h, j * h));
			}
		}

		if (iter % 2 == 0) {
			MPI_Waitall(2, reqSend1, MPI_STATUSES_IGNORE);
			MPI_Waitall(2, reqRecv1, MPI_STATUSES_IGNORE);
		} else {
			MPI_Waitall(2, reqSend2, MPI_STATUSES_IGNORE);
			MPI_Waitall(2, reqRecv2, MPI_STATUSES_IGNORE);
		}

		if (myid != 0) {
			for (int j = 1; j < n - 1; ++j) {  // с учетом верхней
				localsolution[0 * n + j] =
					q * (localsolutionOld[n + j] + lineabove[j] +
						 localsolutionOld[0 * n + (j + 1)] +
						 localsolutionOld[0 * n + (j - 1)] +
						 hh * f((0 + myOffset) * h, j * h));
			}
		}
		if (myid != np - 1) {
			for (int j = 1; j < n - 1; ++j) {  // с учетом нижней
				localsolution[(sizeparts - 1) * n + j] =
					q *
					(linebelow[j] + localsolutionOld[(sizeparts - 2) * n + j] +
					 localsolutionOld[(sizeparts - 1) * n + (j + 1)] +
					 localsolutionOld[(sizeparts - 1) * n + (j - 1)] +
					 hh * f(((sizeparts - 1) + myOffset) * h, j * h));
			}
		}

		double localErr = error(localsolution, localsolutionOld);
		MPI_Allreduce(&localErr, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	} while (err > eps);
	t1 += MPI_Wtime();


	if (myid == 0) {
		for (auto& x : locSize) x *= n;
		for (auto& x : offset) x *= n;
	}
	// сбор результатов
	MPI_Gatherv(localsolution.data(), sizeparts * n, MPI_DOUBLE,
				solution.data(), locSize.data(), offset.data(), MPI_DOUBLE, 0,
				MPI_COMM_WORLD);

	if (myid == 0) {
		double norm = norm1(solution);
		cout << "---Jacobi Send_Init&Recv_Init" << endl;
		cout << "n of iterations: " << iter << endl;
		cout << "Discrepancy=" << err << endl;

		cout << "time=" << t1 << endl;
		cout << endl;
	}
	delete[] reqSend1;
	delete[] reqRecv1;
	delete[] reqSend2;
	delete[] reqRecv2;
}


// Метод красных - черных итераций
void RedBlack1(int myid, int np) {
	double err = 0.0;
	int iteration = 0;
	vector<double> solution;
	vector<int> locSize, offset;

	if (myid == 0) {
		solution.resize(n * n);
		offset.resize(np);
		locSize.resize(np, 0);

		for (int p = 0; p < np - 1; ++p) {
			locSize[p] = n / np;
			offset[p + 1] = offset[p] + locSize[p];
		}
		locSize[np - 1] = n / np + n % np;
	}
	int sizeparts;	// размеры массивов частей решений для каждого из
					// процессоров
	int myOffset;

	MPI_Scatter(locSize.data(), 1, MPI_INT, &sizeparts, 1, MPI_INT, 0,
				MPI_COMM_WORLD);
	vector<double> localsolution(
		sizeparts * n, 0.0);  // локальное решение для каждого из процессоров
	vector<double> solutionold(sizeparts * n, 0.0);

	MPI_Scatter(offset.data(), 1, MPI_INT, &myOffset, 1, MPI_INT, 0,
				MPI_COMM_WORLD);


	vector<double> lineabove(n), linebelow(n);
	double t1 = -MPI_Wtime();
	// тут решение системы для каждого процессора
	double solNorm, diffNorm, solution_norm, difference_norm, err_local;

	int source = myid ? myid - 1 : np - 1;
	int dest = (myid != (np - 1)) ? myid + 1 : 0;

	int sendcount = (myid != (np - 1)) ? n : 0;
	int recvcount = myid ? n : 0;

	do {
		++iteration;
		// 1. перессылка данных (вниз)
		if (np > 1) {
			MPI_Send(solutionold.data() + (sizeparts - 1) * n, sendcount,
					 MPI_DOUBLE, dest, 111, MPI_COMM_WORLD);
			MPI_Recv(lineabove.data(), recvcount, MPI_DOUBLE, source, 111,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

			MPI_Send(solutionold.data(), recvcount, MPI_DOUBLE, source, 222,
					 MPI_COMM_WORLD);
			MPI_Recv(linebelow.data(), sendcount, MPI_DOUBLE, dest, 222,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}

		// 2. вычисление
		// красные
		for (int i = 1; i < sizeparts - 1; ++i) {
			for (int j = 1 + (i + 1) % 2; j < n - 1; j += 2) {
				localsolution[i * n + j] =
					q * (solutionold[(i + 1) * n + j] +
						 solutionold[(i - 1) * n + j] +
						 solutionold[i * n + (j + 1)] +
						 solutionold[i * n + (j - 1)] +
						 hh * f((i + myOffset) * h, j * h));
			}
		}

		if (myid != 0) {
			// красные i = 0
			for (int j = 2; j < n - 1; j += 2) {
				localsolution[j] =
					q *
					(solutionold[n + j] + lineabove[j] + solutionold[j + 1] +
					 solutionold[j - 1] + hh * f(myOffset * h, j * h));
			}
		}

		if (myid != np - 1) {
			// красные i = sizeparts - 1
			for (int j = 1 + sizeparts % 2; j < n - 1; j += 2) {
				localsolution[(sizeparts - 1) * n + j] =
					q * (linebelow[j] + solutionold[(sizeparts - 2) * n + j] +
						 solutionold[(sizeparts - 1) * n + (j + 1)] +
						 solutionold[(sizeparts - 1) * n + (j - 1)] +
						 hh * f((sizeparts - 1 + myOffset) * h, j * h));
			}
		}

		// черные
		for (int i = 1; i < sizeparts - 1; ++i) {
			for (int j = 1 + i % 2; j < n - 1; j += 2) {
				localsolution[i * n + j] =
					q * (localsolution[(i + 1) * n + j] +
						 localsolution[(i - 1) * n + j] +
						 localsolution[i * n + (j + 1)] +
						 localsolution[i * n + (j - 1)] +
						 hh * f((i + myOffset) * h, j * h));
			}
		}

		if (np > 1) {
			MPI_Send(localsolution.data() + (sizeparts - 1) * n, sendcount,
					 MPI_DOUBLE, dest, 111, MPI_COMM_WORLD);
			MPI_Recv(lineabove.data(), recvcount, MPI_DOUBLE, source, 111,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

			MPI_Send(localsolution.data(), recvcount, MPI_DOUBLE, source, 222,
					 MPI_COMM_WORLD);
			MPI_Recv(linebelow.data(), sendcount, MPI_DOUBLE, dest, 222,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}

		if (myid != 0) {
			// черные i = 0
			for (int j = 1; j < n - 1; j += 2) {
				localsolution[j] =
					q * (localsolution[n + j] + lineabove[j] +
						 localsolution[j + 1] + localsolution[j - 1] +
						 hh * f(myOffset * h, j * h));
			}
		}
		if (myid != np - 1) {
			// черные i = sizeparts - 1
			for (int j = 1 + (sizeparts - 1) % 2; j < n - 1; j += 2) {
				localsolution[(sizeparts - 1) * n + j] =
					q * (linebelow[j] + localsolution[(sizeparts - 2) * n + j] +
						 localsolution[(sizeparts - 1) * n + (j + 1)] +
						 localsolution[(sizeparts - 1) * n + (j - 1)] +
						 hh * f((sizeparts - 1 + myOffset) * h, j * h));
			}
		}


		double localErr = error(localsolution, solutionold);
		MPI_Allreduce(&localErr, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		swap(localsolution, solutionold);
	} while (err > eps);
	t1 += MPI_Wtime();

	// изменяем массив locSize, чтобы он хранил количество элементов для каждого
	// процессора. До этого хранил количество обрабатываемых СТРОК
	if (myid == 0) {
		for (auto& x : locSize) x *= n;
		for (auto& x : offset) x *= n;
	}

	MPI_Gatherv(solutionold.data(), sizeparts * n, MPI_DOUBLE, solution.data(),
				locSize.data(), offset.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (myid == 0) {
		double norm = norm1(solution);
		cout << "---Red - Black MPI_Send&MPI_Recv---" << endl;
		cout << "n of iterations: " << iteration << endl;

		cout << "Discrepancy=" << norm << endl;
		cout << "time=" << t1 << endl << endl;
	}
}


void RedBlack2(int myid, int np) {
	double err = 0.0;
	int iteration = 0;
	vector<double> solution;
	vector<int> locSize, offset;

	if (myid == 0) {
		solution.resize(n * n);
		offset.resize(np);
		locSize.resize(np, 0);

		for (int p = 0; p < np - 1; ++p) {
			locSize[p] = n / np;
			offset[p + 1] = offset[p] + locSize[p];
		}
		locSize[np - 1] = n / np + n % np;
	}
	int sizeparts;	// размеры массивов частей решений для каждого из
					// процессоров
	int myOffset;

	MPI_Scatter(locSize.data(), 1, MPI_INT, &sizeparts, 1, MPI_INT, 0,
				MPI_COMM_WORLD);
	vector<double> localsolution(
		sizeparts * n, 0.0);  // локальное решение для каждого из процессоров
	vector<double> solutionold(sizeparts * n, 0.0);

	MPI_Scatter(offset.data(), 1, MPI_INT, &myOffset, 1, MPI_INT, 0,
				MPI_COMM_WORLD);


	vector<double> lineabove(n), linebelow(n);
	double t1 = -MPI_Wtime();
	// тут решение системы для каждого процессора
	double solNorm, diffNorm, solution_norm, difference_norm, err_local;

	do {
		++iteration;
		// 1. Передача данных
		int sendCount = (myid != (np - 1)) ? n : 0, recvCount = myid ? n : 0;
		int dest = (myid != (np - 1)) ? myid + 1 : 0,
			source = myid ? myid - 1 : np - 1;	// кому, откуда

		MPI_Sendrecv(solutionold.data() + (sizeparts - 1) * n, sendCount,
					 MPI_DOUBLE, dest, 111, lineabove.data(), recvCount,
					 MPI_DOUBLE, source, 111, MPI_COMM_WORLD,
					 MPI_STATUSES_IGNORE);
		MPI_Sendrecv(solutionold.data(), recvCount, MPI_DOUBLE, source, 222,
					 linebelow.data(), sendCount, MPI_DOUBLE, dest, 222,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		// 2. вычисление
		// красные
		for (int i = 1; i < sizeparts - 1; ++i) {
			for (int j = 1 + (i + 1) % 2; j < n - 1; j += 2) {
				localsolution[i * n + j] =
					q * (solutionold[(i + 1) * n + j] +
						 solutionold[(i - 1) * n + j] +
						 solutionold[i * n + (j + 1)] +
						 solutionold[i * n + (j - 1)] +
						 hh * f((i + myOffset) * h, j * h));
			}
		}

		if (myid != 0) {
			// красные i = 0
			for (int j = 2; j < n - 1; j += 2) {
				localsolution[j] =
					q *
					(solutionold[n + j] + lineabove[j] + solutionold[j + 1] +
					 solutionold[j - 1] + hh * f(myOffset * h, j * h));
			}
		}

		if (myid != np - 1) {
			// красные i = sizeparts - 1
			for (int j = 1 + sizeparts % 2; j < n - 1; j += 2) {
				localsolution[(sizeparts - 1) * n + j] =
					q * (linebelow[j] + solutionold[(sizeparts - 2) * n + j] +
						 solutionold[(sizeparts - 1) * n + (j + 1)] +
						 solutionold[(sizeparts - 1) * n + (j - 1)] +
						 hh * f((sizeparts - 1 + myOffset) * h, j * h));
			}
		}

		// черные
		for (int i = 1; i < sizeparts - 1; ++i) {
			for (int j = 1 + i % 2; j < n - 1; j += 2) {
				localsolution[i * n + j] =
					q * (localsolution[(i + 1) * n + j] +
						 localsolution[(i - 1) * n + j] +
						 localsolution[i * n + (j + 1)] +
						 localsolution[i * n + (j - 1)] +
						 hh * f((i + myOffset) * h, j * h));
			}
		}

		MPI_Sendrecv(localsolution.data() + (sizeparts - 1) * n, sendCount,
					 MPI_DOUBLE, dest, 111, lineabove.data(), recvCount,
					 MPI_DOUBLE, source, 111, MPI_COMM_WORLD,
					 MPI_STATUSES_IGNORE);
		MPI_Sendrecv(localsolution.data(), recvCount, MPI_DOUBLE, source, 222,
					 linebelow.data(), sendCount, MPI_DOUBLE, dest, 222,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

		if (myid != 0) {
			// черные i = 0
			for (int j = 1; j < n - 1; j += 2) {
				localsolution[j] =
					q * (localsolution[n + j] + lineabove[j] +
						 localsolution[j + 1] + localsolution[j - 1] +
						 hh * f(myOffset * h, j * h));
			}
		}
		if (myid != np - 1) {
			// черные i = sizeparts - 1
			for (int j = 1 + (sizeparts - 1) % 2; j < n - 1; j += 2) {
				localsolution[(sizeparts - 1) * n + j] =
					q * (linebelow[j] + localsolution[(sizeparts - 2) * n + j] +
						 localsolution[(sizeparts - 1) * n + (j + 1)] +
						 localsolution[(sizeparts - 1) * n + (j - 1)] +
						 hh * f((sizeparts - 1 + myOffset) * h, j * h));
			}
		}

		// 3. посчитать локальные ошибки, выбрать среди них максимальную,
		// присвоить это err, распространить err среди всех

		double localErr = error(localsolution, solutionold);
		MPI_Allreduce(&localErr, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		swap(localsolution, solutionold);
	} while (err > eps);

	t1 += MPI_Wtime();

	// изменяем массив locSize, чтобы он хранил количество элементов для каждого
	// процессора. До этого хранил количество обрабатываемых СТРОК
	if (myid == 0) {
		for (auto& x : locSize) x *= n;
		for (auto& x : offset) x *= n;
	}

	MPI_Gatherv(solutionold.data(), sizeparts * n, MPI_DOUBLE, solution.data(),
				locSize.data(), offset.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (myid == 0) {
		double norm = norm1(solution);
		cout << "---Red - Black MPI_Sendrecv---" << endl;
		cout << "n of iterations: " << iteration << endl;

		cout << "Discrepancy=" << norm << endl;
		cout << "time=" << t1 << endl << endl;
	}
}


void RedBlack3(int myid, int np) {
	double err = 0.0;
	int iter = 0;
	vector<double> solution;
	vector<int> locSize, offset;

	if (myid == 0) {
		solution.resize(n * n);
		offset.resize(np);
		locSize.resize(np, 0);

		for (int p = 0; p < np - 1; ++p) {
			locSize[p] = n / np;
			offset[p + 1] = offset[p] + locSize[p];
		}
		locSize[np - 1] = n / np + n % np;
	}
	int sizeparts;	// размеры массивов частей решений для каждого из
					// процессоров
	int myOffset;

	MPI_Scatter(locSize.data(), 1, MPI_INT, &sizeparts, 1, MPI_INT, 0,
				MPI_COMM_WORLD);
	vector<double> localsolution(
		sizeparts * n, 0.0);  // локальное решение для каждого из процессоров
	vector<double> localsolutionOld(sizeparts * n, 0.0);

	MPI_Scatter(offset.data(), 1, MPI_INT, &myOffset, 1, MPI_INT, 0,
				MPI_COMM_WORLD);


	vector<double> lineabove(n), linebelow(n);
	double t1 = -MPI_Wtime();
	// тут решение системы для каждого процессора
	double solNorm, diffNorm, solution_norm, difference_norm, err_local;


	MPI_Request *reqSend1, *reqRecv1, *reqSend2,
		*reqRecv2;	// массивы из запросов на отправку/получение
	int topSize, btmSize, idTop, idBtm;

	idTop = (myid != 0) ? myid - 1 : np - 1;
	idBtm = (myid != np - 1) ? myid + 1 : 0;

	topSize = (myid != 0) ? n : 0;
	btmSize = (myid != np - 1) ? n : 0;

	reqSend1 = new MPI_Request[2];
	reqRecv1 = new MPI_Request[2];
	reqSend2 = new MPI_Request[2];
	reqRecv2 = new MPI_Request[2];

	//////////////
	MPI_Send_init(localsolutionOld.data(), topSize, MPI_DOUBLE, idTop, 111,
				  MPI_COMM_WORLD, reqSend1);
	MPI_Recv_init(linebelow.data(), btmSize, MPI_DOUBLE, idBtm, MPI_ANY_TAG,
				  MPI_COMM_WORLD, reqRecv1);

	MPI_Send_init(localsolutionOld.data() + (sizeparts - 1) * n, btmSize,
				  MPI_DOUBLE, idBtm, 111, MPI_COMM_WORLD, reqSend1 + 1);
	MPI_Recv_init(lineabove.data(), topSize, MPI_DOUBLE, idTop, MPI_ANY_TAG,
				  MPI_COMM_WORLD, reqRecv1 + 1);

	MPI_Send_init(localsolution.data(), topSize, MPI_DOUBLE, idTop, 111,
				  MPI_COMM_WORLD, reqSend2);
	MPI_Recv_init(linebelow.data(), btmSize, MPI_DOUBLE, idBtm, MPI_ANY_TAG,
				  MPI_COMM_WORLD, reqRecv2);

	MPI_Send_init(localsolution.data() + (sizeparts - 1) * n, btmSize,
				  MPI_DOUBLE, idBtm, 111, MPI_COMM_WORLD, reqSend2 + 1);
	MPI_Recv_init(lineabove.data(), topSize, MPI_DOUBLE, idTop, MPI_ANY_TAG,
				  MPI_COMM_WORLD, reqRecv2 + 1);
	/////////////

	do {
		++iter;
		swap(localsolution, localsolutionOld);

		if (iter % 2 == 0) {
			MPI_Startall(2, reqSend1);
			MPI_Startall(2, reqRecv1);
		} else {
			MPI_Startall(2, reqSend2);
			MPI_Startall(2, reqRecv2);
		}
		// вычисление во внутренних узлах, красные
		for (int i = 1; i < sizeparts - 1; ++i) {
			for (int j = 1 + (i + 1) % 2; j < n - 1; j += 2) {
				localsolution[i * n + j] =
					q * (localsolutionOld[(i + 1) * n + j] +
						 localsolutionOld[(i - 1) * n + j] +
						 localsolutionOld[i * n + (j + 1)] +
						 localsolutionOld[i * n + (j - 1)] +
						 hh * f((i + myOffset) * h, j * h));
			}
		}

		if (iter % 2 == 0) {
			MPI_Waitall(2, reqSend1, MPI_STATUSES_IGNORE);
			MPI_Waitall(2, reqRecv1, MPI_STATUSES_IGNORE);
		} else {
			MPI_Waitall(2, reqSend2, MPI_STATUSES_IGNORE);
			MPI_Waitall(2, reqRecv2, MPI_STATUSES_IGNORE);
		}

		if (myid != 0) {
			// красные i = 0
			for (int j = 2; j < n - 1; j += 2) {
				localsolution[j] =
					q * (localsolutionOld[n + j] + lineabove[j] +
						 localsolutionOld[j + 1] + localsolutionOld[j - 1] +
						 hh * f(myOffset * h, j * h));
			}
		}
		if (myid != np - 1) {
			// красные i = sizeparts - 1
			for (int j = 1 + sizeparts % 2; j < n - 1; j += 2) {
				localsolution[(sizeparts - 1) * n + j] =
					q *
					(linebelow[j] + localsolutionOld[(sizeparts - 2) * n + j] +
					 localsolutionOld[(sizeparts - 1) * n + (j + 1)] +
					 localsolutionOld[(sizeparts - 1) * n + (j - 1)] +
					 hh * f((sizeparts - 1 + myOffset) * h, j * h));
			}
		}
		////
		if (iter % 2 == 0) {
			MPI_Startall(2, reqSend2);
			MPI_Startall(2, reqRecv2);
		} else {
			MPI_Startall(2, reqSend1);
			MPI_Startall(2, reqRecv1);
		}
		// черные
		for (int i = 1; i < sizeparts - 1; ++i) {
			for (int j = 1 + i % 2; j < n - 1; j += 2) {
				localsolution[i * n + j] =
					q * (localsolution[(i + 1) * n + j] +
						 localsolution[(i - 1) * n + j] +
						 localsolution[i * n + (j + 1)] +
						 localsolution[i * n + (j - 1)] +
						 hh * f((i + myOffset) * h, j * h));
			}
		}

		if (iter % 2 == 0) {
			MPI_Waitall(2, reqSend2, MPI_STATUSES_IGNORE);
			MPI_Waitall(2, reqRecv2, MPI_STATUSES_IGNORE);
		} else {
			MPI_Waitall(2, reqSend1, MPI_STATUSES_IGNORE);
			MPI_Waitall(2, reqRecv1, MPI_STATUSES_IGNORE);
		}
		if (myid != 0) {
			// черные i = 0
			for (int j = 1; j < n - 1; j += 2) {
				localsolution[j] =
					q * (localsolution[n + j] + lineabove[j] +
						 localsolution[j + 1] + localsolution[j - 1] +
						 hh * f(myOffset * h, j * h));
			}
		}
		if (myid != np - 1) {
			// черные i = sizeparts - 1
			for (int j = 1 + (sizeparts - 1) % 2; j < n - 1; j += 2) {
				localsolution[(sizeparts - 1) * n + j] =
					q * (linebelow[j] + localsolution[(sizeparts - 2) * n + j] +
						 localsolution[(sizeparts - 1) * n + (j + 1)] +
						 localsolution[(sizeparts - 1) * n + (j - 1)] +
						 hh * f((sizeparts - 1 + myOffset) * h, j * h));
			}
		}

		double localErr = error(localsolution, localsolutionOld);
		MPI_Allreduce(&localErr, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	} while (err > eps);

	t1 += MPI_Wtime();

	// изменяем массив locSize, чтобы он хранил количество элементов для каждого
	// процессора. До этого хранил количество обрабатываемых СТРОК
	if (myid == 0) {
		for (auto& x : locSize) x *= n;
		for (auto& x : offset) x *= n;
	}

	MPI_Gatherv(localsolution.data(), sizeparts * n, MPI_DOUBLE,
				solution.data(), locSize.data(), offset.data(), MPI_DOUBLE, 0,
				MPI_COMM_WORLD);

	if (myid == 0) {
		double norm = norm1(solution);
		cout << "---Red - Black Send_Init&Recv_Init" << endl;

		cout << "n of iterations: " << iter << endl;

		cout << "Discrepancy=" << norm << endl;
		cout << "time=" << t1 << endl << endl;
	}
	delete[] reqSend1;
	delete[] reqRecv1;
	delete[] reqSend2;
	delete[] reqRecv2;
}


int main(int argc, char** argv) {
	int processors, numOfProcess;
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD,
				  &numOfProcess);  // для определения ранга процесса
	MPI_Comm_size(MPI_COMM_WORLD,
				  &processors);	 // определение количества процессов в
								 // выполняемой параллельной программе
	if (numOfProcess == 0)
		cout << "matrix size=" << n << endl
			 << "num of procs=" << processors << endl
			 << endl;

	Jacobi1(numOfProcess, processors);

	Jacobi2(numOfProcess, processors);

	Jacobi3(numOfProcess, processors);


	RedBlack1(numOfProcess, processors);

	RedBlack2(numOfProcess, processors);

	RedBlack3(numOfProcess, processors);


	MPI_Finalize();

	return 0;
}
