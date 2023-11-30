#define _USE_MATH_DEFINES
#include <mpi.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

void Jacobi_Send_Recv(int id, int size);
void Jacobi_SendRecv(int id, int size);
void Jacobi_ISend_IRecv(int id, int size);

void Zeydel_Send_Recv(int id, int size);
void Zeydel_SendRecv(int id, int size);
void Zeydel_ISend_IRecv(int id, int size);

const double eps = 1e-6;  // Инициализация подсистемы MPI
const int n = 10000;
const double h = 1.0 / (double)(n - 1);
const double k = (double)n;
double q = 1.0 / (4.0 + k * k * h * h);

double u(double x, double y) { return (1.0 - x) * x * sin(M_PI * y); }

double sqr(const double a) { return (a * a); }

double f(double x, double y) {
	return (2.0 * sin(M_PI * y) + k * k * (1.0 - x) * x * sin(M_PI * y) +
			M_PI * M_PI * (1.0 - x) * x * sin(M_PI * y));
}

double f_Real(double x, double y) { return (1.0 - x) * x * sin(M_PI * y); }

double norm(const vector<double>& y) {
	double norma = 0.0;
	for (int i = 0; i < y.size(); ++i) norma += y[i] * y[i];
	return sqrt(norma);
}

void vec_u(std::vector<double>& y_real) {
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j) y_real[i * n + j] = u(j * h, i * h);
}

double error(const vector<double>& y, const vector<double>& u) {
	double norma = 0.0;
	for (int i = 0; i < n * n; ++i) norma += (y[i] - u[i]) * (y[i] - u[i]);
	return sqrt(norma);
}

double check_norm(const vector<double>& a, const vector<double>& b) {
	double max = 0.0;
	double tmp = 0.0;

	for (int i = 0; i < a.size(); ++i) {
		tmp = fabs(a[i] - b[i]);
		if (tmp > max) max = tmp;
	}

	return max;
}


int main(int argc, char** argv) {
	int id;
	int size;
	int m = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,
				  &size);  // указывает количество процессов (Получаем
						   // количество запущенных процессов)
	MPI_Comm_rank(MPI_COMM_WORLD,
				  &id);	 // ранг процессора, возвращает указатель на
						 // идентификатор вызывающего процессора

	if (id == 0)
		cout << "\n Parametres: n = " << n << ", k = " << k
			 << ", size = " << size;

	if (id == 0) cout << "\n\n ------ Jacobi ------";
	Jacobi_Send_Recv(id, size);
	Jacobi_SendRecv(id, size);
	Jacobi_ISend_IRecv(id, size);

	if (id == 0) cout << "\n\n ------ Zeydel ------";
	Zeydel_Send_Recv(id, size);
	Zeydel_SendRecv(id, size);
	Zeydel_ISend_IRecv(id, size);

	MPI_Finalize();

	return 0;
}

void str_split(int id, int size, int& str_local, int& nums_local,
			   vector<int>& str_per_proc, vector<int>& nums_start) {
	str_per_proc.resize(size, n / size);
	nums_start.resize(size, 0);

	for (int i = 0; i < n % size; ++i) ++str_per_proc[i];

	for (int i = 1; i < size; ++i)
		nums_start[i] = nums_start[i - 1] + str_per_proc[i - 1];

	MPI_Scatter(str_per_proc.data(), 1, MPI_INT, &str_local, 1, MPI_INT, 0,
				MPI_COMM_WORLD);  // распределяем данные по группам
	MPI_Scatter(nums_start.data(), 1, MPI_INT, &nums_local, 1, MPI_INT, 0,
				MPI_COMM_WORLD);
}

void Jacobi_Send_Recv(int id, int size) {
	vector<int> str_per_proc, nums_start;
	int str_local, nums_local;
	double norm_local, norm_err;

	str_split(id, size, str_local, nums_local, str_per_proc, nums_start);

	vector<double> y_local(str_local * n);
	vector<double> y_next_top(n);
	vector<double> y_prev_low(n);

	int source_proc = id ? id - 1 : size - 1;
	int dest_proc = (id != (size - 1)) ? id + 1 : 0;

	int scount = (id != (size - 1)) ? n : 0;
	int rcount = id ? n : 0;

	vector<double> y;
	if (id == 0) y.resize(n * n);

	double t1 = -MPI_Wtime();

	int it = 0;
	bool flag = true;
	vector<double> temp(y_local.size());
	while (flag) {
		it++;
		// cout << "\n it = " << it;

		std::swap(temp, y_local);

		// пересылаем нижние строки всеми процессами кроме последнего
		MPI_Send(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE,
				 dest_proc, 42, MPI_COMM_WORLD);
		MPI_Recv(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 42,
				 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

		// пересылаем верхние строки всеми процессами кроме нулевого
		MPI_Send(temp.data(), rcount, MPI_DOUBLE, source_proc, 46,
				 MPI_COMM_WORLD);
		MPI_Recv(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 46,
				 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

		/* пересчитываем все строки в полосе кроме верхней и нижней */
		for (int i = 1; i < str_local - 1; ++i)
			for (int j = 1; j < n - 1; ++j)
				y_local[i * n + j] =
					(temp[(i + 1) * n + j] + temp[(i - 1) * n + j] +
					 temp[i * n + (j + 1)] + temp[i * n + (j - 1)] +
					 h * h * f((nums_local + i) * h, j * h)) *
					q;

		/* пересчитываем верхние строки */
		if (id != 0)
			for (int j = 1; j < n - 1; ++j)
				y_local[j] = (temp[n + j] + y_prev_low[j] + temp[j + 1] +
							  temp[j - 1] + h * h * f(nums_local * h, j * h)) *
							 q;

		/* пересчитываем нижние строки */
		if (id != size - 1)
			for (int j = 1; j < n - 1; ++j)
				y_local[(str_local - 1) * n + j] =
					(y_next_top[j] + temp[(str_local - 2) * n + j] +
					 temp[(str_local - 1) * n + (j + 1)] +
					 temp[(str_local - 1) * n + (j - 1)] +
					 h * h * f((nums_local + (str_local - 1)) * h, j * h)) *
					q;

		norm_local = check_norm(temp, y_local);

		MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX,
					  MPI_COMM_WORLD);	// Объединяем значения из всех процессов

		if (norm_err < eps) flag = false;
	}

	t1 += MPI_Wtime();

	for (int i = 0; i < size; ++i) {
		str_per_proc[i] *= n;
		nums_start[i] *= n;
	}

	MPI_Gatherv(y_local.data(), str_local * n, MPI_DOUBLE, y.data(),
				str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0,
				MPI_COMM_WORLD);  // Собираем данные переменных из всех членов
								  // группы в один элемент

	if (id == 0) {
		// точное решение
		vector<double> check(n * n);
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j) check[i * n + j] = u(i * h, j * h);

		cout << "\n\n Якоби Send + Recv";
		cout << "\n\t norm: " << check_norm(y, check);
		cout << "\n\t iterations: " << it;
		printf("\n\t time: %.4f", t1);
	}
}

void Zeydel_Send_Recv(int id, int size) {
	vector<int> str_per_proc, nums_start;
	int str_local, nums_local;
	double norm_local, norm_err;

	str_split(id, size, str_local, nums_local, str_per_proc, nums_start);

	vector<double> y_local(str_local * n);
	vector<double> y_next_top(n);
	vector<double> y_prev_low(n);

	int source_proc = id ? id - 1 : size - 1;		  // id == 0 = size - 1
	int dest_proc = (id != (size - 1)) ? id + 1 : 0;  // у id == size - 1 = 0

	int scount = (id != (size - 1)) ? n : 0;  // у id == size - 1 = 0
	int rcount = id ? n : 0;				  // id == 0 = 0

	vector<double> y;
	if (id == 0) y.resize(n * n);

	double t1 = -MPI_Wtime();

	int it = 0;
	bool flag = true;
	vector<double> temp(y_local.size());
	while (flag) {
		it++;
		// cout << "\n it = " << it;

		std::swap(temp, y_local);

		// y_local = temp;

		// пересылаем нижние строки всеми процессами кроме последнего
		MPI_Send(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE,
				 dest_proc, 42, MPI_COMM_WORLD);
		MPI_Recv(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 42,
				 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

		// пересылаем верхние строки всеми процессами кроме нулевого
		MPI_Send(temp.data(), rcount, MPI_DOUBLE, source_proc, 46,
				 MPI_COMM_WORLD);
		MPI_Recv(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 46,
				 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

		// внутренние строки (красные)
		for (int i = 1; i < str_local - 1; ++i)
			for (int j = ((i + 1) % 2 + 1); j < n - 1; j += 2)
				y_local[i * n + j] =
					(temp[(i + 1) * n + j] + temp[(i - 1) * n + j] +
					 temp[i * n + (j + 1)] + temp[i * n + (j - 1)] +
					 h * h * f((nums_local + i) * h, j * h)) *
					q;

		// верхние строки (красные)
		if (id != 0)
			for (int j = 2; j < n - 1; j += 2)
				y_local[j] = (temp[n + j] + y_prev_low[j] + temp[j + 1] +
							  temp[j - 1] + h * h * f(nums_local * h, j * h)) *
							 q;

		// нижние строки (красные)
		if (id != size - 1)
			for (int j = 1 + str_local % 2; j < n - 1; j += 2)
				y_local[(str_local - 1) * n + j] =
					(y_next_top[j] + temp[(str_local - 2) * n + j] +
					 temp[(str_local - 1) * n + (j + 1)] +
					 temp[(str_local - 1) * n + (j - 1)] +
					 h * h * f((nums_local + (str_local - 1)) * h, j * h)) *
					q;

		MPI_Barrier;
		// пересылаем нижние строки всеми процессами кроме последнего
		MPI_Send(y_local.data() + (str_local - 1) * n, scount, MPI_DOUBLE,
				 dest_proc, 42, MPI_COMM_WORLD);
		MPI_Recv(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 42,
				 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

		// пересылаем верхние строки всеми процессами кроме нулевого
		MPI_Send(y_local.data(), rcount, MPI_DOUBLE, source_proc, 46,
				 MPI_COMM_WORLD);
		MPI_Recv(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 46,
				 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);


		// внутренние строки (чёрные)
		for (int i = 1; i < str_local - 1; ++i)
			for (int j = (i % 2 + 1); j < n - 1; j += 2)
				y_local[i * n + j] =
					(y_local[(i + 1) * n + j] + y_local[(i - 1) * n + j] +
					 y_local[i * n + (j + 1)] + y_local[i * n + (j - 1)] +
					 h * h * f((nums_local + i) * h, j * h)) *
					q;

		// верхние строки (чёрные)
		if (id != 0)
			for (int j = 1; j < n - 1; j += 2)
				y_local[j] =
					(y_local[n + j] + y_prev_low[j] + y_local[j + 1] +
					 y_local[j - 1] + h * h * f(nums_local * h, j * h)) *
					q;

		// нижние строки (чёрные)
		if (id != size - 1)
			for (int j = 1 + (str_local - 1) % 2; j < n - 1; j += 2)
				y_local[(str_local - 1) * n + j] =
					(y_next_top[j] + y_local[(str_local - 2) * n + j] +
					 y_local[(str_local - 1) * n + (j + 1)] +
					 y_local[(str_local - 1) * n + (j - 1)] +
					 h * h * f((nums_local + (str_local - 1)) * h, j * h)) *
					q;

		norm_local = check_norm(temp, y_local);

		MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX,
					  MPI_COMM_WORLD);

		if (norm_err < eps) flag = false;
	}

	t1 += MPI_Wtime();

	for (int i = 0; i < size; ++i) {
		str_per_proc[i] *= n;
		nums_start[i] *= n;
	}

	// MPI_Gather(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_local
	// * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gatherv(y_local.data(), str_local * n, MPI_DOUBLE, y.data(),
				str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0,
				MPI_COMM_WORLD);

	if (id == 0) {
		// точное решение
		vector<double> check(n * n);
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j) check[i * n + j] = u(i * h, j * h);

		cout << "\n\n Зейдель Send + Recv";
		cout << "\n\t norm: " << check_norm(y, check);
		cout << "\n\t iterations: " << it;
		printf("\n\t time: %.4f", t1);
	}
}

void Jacobi_SendRecv(int id, int size) {
	vector<int> str_per_proc, nums_start;
	int str_local, nums_local;
	double norm_local, norm_err;

	str_split(id, size, str_local, nums_local, str_per_proc, nums_start);

	vector<double> y_local(str_local * n);
	vector<double> y_next_top(n);
	vector<double> y_prev_low(n);

	vector<double> y;
	if (id == 0) y.resize(n * n);

	double t1 = -MPI_Wtime();

	int source_proc = id ? id - 1 : size - 1;
	int dest_proc = (id != (size - 1)) ? id + 1 : 0;

	int scount = (id != (size - 1)) ? n : 0;
	int rcount = id ? n : 0;

	int it = 0;
	bool flag = true;
	vector<double> temp(y_local.size());
	while (flag) {
		it++;

		std::swap(temp, y_local);

		MPI_Sendrecv(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE,
					 dest_proc, 42, y_prev_low.data(), rcount, MPI_DOUBLE,
					 source_proc, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		MPI_Sendrecv(temp.data(), rcount, MPI_DOUBLE, source_proc, 46,
					 y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 46,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

		/* пересчитываем все строки в полосе кроме верхней и нижней */
		for (int i = 1; i < str_local - 1; ++i)
			for (int j = 1; j < n - 1; ++j)
				y_local[i * n + j] =
					(temp[(i + 1) * n + j] + temp[(i - 1) * n + j] +
					 temp[i * n + (j + 1)] + temp[i * n + (j - 1)] +
					 h * h * f((nums_local + i) * h, j * h)) *
					q;

		/* пересчитываем верхние строки */
		if (id != 0)
			for (int j = 1; j < n - 1; ++j)
				y_local[j] = (temp[n + j] + y_prev_low[j] + temp[j + 1] +
							  temp[j - 1] + h * h * f(nums_local * h, j * h)) *
							 q;

		/* пересчитываем нижние строки */
		if (id != size - 1)
			for (int j = 1; j < n - 1; ++j)
				y_local[(str_local - 1) * n + j] =
					(y_next_top[j] + temp[(str_local - 2) * n + j] +
					 temp[(str_local - 1) * n + (j + 1)] +
					 temp[(str_local - 1) * n + (j - 1)] +
					 h * h * f((nums_local + (str_local - 1)) * h, j * h)) *
					q;

		norm_local = check_norm(temp, y_local);

		MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX,
					  MPI_COMM_WORLD);

		if (norm_err < eps) flag = false;
	}

	t1 += MPI_Wtime();

	for (int i = 0; i < size; ++i) {
		str_per_proc[i] *= n;
		nums_start[i] *= n;
	}

	// MPI_Gather(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_local
	// * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gatherv(y_local.data(), str_local * n, MPI_DOUBLE, y.data(),
				str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0,
				MPI_COMM_WORLD);

	if (id == 0) {
		// точное решение
		vector<double> check(n * n);
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j) check[i * n + j] = u(i * h, j * h);

		cout << "\n\n Якоби SendRecv";
		cout << "\n\t norm: " << check_norm(y, check);
		cout << "\n\t iterations: " << it;
		printf("\n\t time: %.4f", t1);
	}
}
void Zeydel_SendRecv(int id, int size) {
	vector<int> str_per_proc, nums_start;
	int str_local, nums_local;
	double norm_local, norm_err;

	str_split(id, size, str_local, nums_local, str_per_proc, nums_start);

	vector<double> y_local(str_local * n);
	vector<double> y_next_top(n);
	vector<double> y_prev_low(n);

	vector<double> y;
	if (id == 0) y.resize(n * n);

	double t1 = -MPI_Wtime();

	int source_proc = id ? id - 1 : size - 1;
	int dest_proc = (id != (size - 1)) ? id + 1 : 0;

	int scount = (id != (size - 1)) ? n : 0;
	int rcount = id ? n : 0;

	int it = 0;
	bool flag = true;
	vector<double> temp(y_local.size());
	while (flag) {
		it++;

		std::swap(temp, y_local);

		// пересылаем нижние и верхние строки
		MPI_Sendrecv(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE,
					 dest_proc, 42, y_prev_low.data(), rcount, MPI_DOUBLE,
					 source_proc, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		MPI_Sendrecv(temp.data(), rcount, MPI_DOUBLE, source_proc, 46,
					 y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 46,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

		// внутренние строки (красные)
		for (int i = 1; i < str_local - 1; ++i)
			for (int j = ((i + 1) % 2 + 1); j < n - 1; j += 2)
				y_local[i * n + j] =
					(temp[(i + 1) * n + j] + temp[(i - 1) * n + j] +
					 temp[i * n + (j + 1)] + temp[i * n + (j - 1)] +
					 h * h * f((nums_local + i) * h, j * h)) *
					q;

		// верхние строки (красные)
		if (id != 0)
			for (int j = 2; j < n - 1; j += 2)
				y_local[j] = (temp[n + j] + y_prev_low[j] + temp[j + 1] +
							  temp[j - 1] + h * h * f(nums_local * h, j * h)) *
							 q;

		// нижние строки (красные)
		if (id != size - 1)
			for (int j = 1 + str_local % 2; j < n - 1; j += 2)
				y_local[(str_local - 1) * n + j] =
					(y_next_top[j] + temp[(str_local - 2) * n + j] +
					 temp[(str_local - 1) * n + (j + 1)] +
					 temp[(str_local - 1) * n + (j - 1)] +
					 h * h * f((nums_local + (str_local - 1)) * h, j * h)) *
					q;

		MPI_Barrier;

		// пересылаем нижние и верхние строки
		MPI_Sendrecv(y_local.data() + (str_local - 1) * n, scount, MPI_DOUBLE,
					 dest_proc, 42, y_prev_low.data(), rcount, MPI_DOUBLE,
					 source_proc, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		MPI_Sendrecv(y_local.data(), rcount, MPI_DOUBLE, source_proc, 46,
					 y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 46,
					 MPI_COMM_WORLD, MPI_STATUSES_IGNORE);


		// внутренние строки (чёрные)
		for (int i = 1; i < str_local - 1; ++i)
			for (int j = (i % 2 + 1); j < n - 1; j += 2)
				y_local[i * n + j] =
					(y_local[(i + 1) * n + j] + y_local[(i - 1) * n + j] +
					 y_local[i * n + (j + 1)] + y_local[i * n + (j - 1)] +
					 h * h * f((nums_local + i) * h, j * h)) *
					q;

		// верхние строки (чёрные)
		if (id != 0)
			for (int j = 1; j < n - 1; j += 2)
				y_local[j] =
					(y_local[n + j] + y_prev_low[j] + y_local[j + 1] +
					 y_local[j - 1] + h * h * f(nums_local * h, j * h)) *
					q;

		// нижние строки (чёрные)
		if (id != size - 1)
			for (int j = 1 + (str_local - 1) % 2; j < n - 1; j += 2)
				y_local[(str_local - 1) * n + j] =
					(y_next_top[j] + y_local[(str_local - 2) * n + j] +
					 y_local[(str_local - 1) * n + (j + 1)] +
					 y_local[(str_local - 1) * n + (j - 1)] +
					 h * h * f((nums_local + (str_local - 1)) * h, j * h)) *
					q;


		norm_local = check_norm(temp, y_local);

		MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX,
					  MPI_COMM_WORLD);

		if (norm_err < eps) flag = false;
	}

	t1 += MPI_Wtime();

	for (int i = 0; i < size; ++i) {
		str_per_proc[i] *= n;
		nums_start[i] *= n;
	}

	MPI_Gatherv(y_local.data(), str_local * n, MPI_DOUBLE, y.data(),
				str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0,
				MPI_COMM_WORLD);

	if (id == 0) {
		// точное решение
		vector<double> check(n * n);
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j) check[i * n + j] = u(i * h, j * h);

		cout << "\n\n Зейдель SendRecv";
		cout << "\n\t norm: " << check_norm(y, check);
		cout << "\n\t iterations: " << it;
		printf("\n\t time: %.4f", t1);
	}
}

void Jacobi_ISend_IRecv(int id, int size) {
	MPI_Request* send_req1;
	MPI_Request* send_req2;
	MPI_Request* recv_req1;
	MPI_Request* recv_req2;

	vector<int> str_per_proc, nums_start;
	int str_local, nums_local;
	double norm_local, norm_err;

	str_split(id, size, str_local, nums_local, str_per_proc, nums_start);

	send_req1 = new MPI_Request[2], recv_req1 = new MPI_Request[2];
	send_req2 = new MPI_Request[2], recv_req2 = new MPI_Request[2];

	vector<double> y_local(str_local * n);
	vector<double> y_next_top(n);
	vector<double> y_prev_low(n);

	int source_proc = id ? id - 1 : size - 1;
	int dest_proc = (id != (size - 1)) ? id + 1 : 0;

	int scount = (id != (size - 1)) ? n : 0;
	int rcount = id ? n : 0;
	vector<double> temp(y_local.size());

	// пересылаем верхние и нижние строки temp
	MPI_Send_init(temp.data(), rcount, MPI_DOUBLE, source_proc, 0,
				  MPI_COMM_WORLD, send_req1);
	MPI_Recv_init(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 1,
				  MPI_COMM_WORLD, recv_req1);

	MPI_Send_init(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE,
				  dest_proc, 1, MPI_COMM_WORLD, send_req1 + 1);
	MPI_Recv_init(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 0,
				  MPI_COMM_WORLD, recv_req1 + 1);

	// пересылаем верхние и нижние строки y_local
	MPI_Send_init(y_local.data(), rcount, MPI_DOUBLE, source_proc, 0,
				  MPI_COMM_WORLD, send_req2);
	MPI_Recv_init(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 1,
				  MPI_COMM_WORLD, recv_req2);

	MPI_Send_init(y_local.data() + (str_local - 1) * n, scount, MPI_DOUBLE,
				  dest_proc, 1, MPI_COMM_WORLD, send_req2 + 1);
	MPI_Recv_init(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 0,
				  MPI_COMM_WORLD, recv_req2 + 1);

	vector<double> y;
	if (id == 0) y.resize(n * n);

	double t1 = -MPI_Wtime();

	int it = 0;
	bool flag = true;
	while (flag) {
		it++;

		std::swap(temp, y_local);

		if (it % 2 == 0) {
			MPI_Startall(2, send_req1);
			MPI_Startall(2, recv_req1);
		} else {
			MPI_Startall(2, send_req2);
			MPI_Startall(2, recv_req2);
		}

		/* пересчитываем все строки в полосе кроме верхней и нижней пока идёт
		 * пересылка */
		for (int i = 1; i < str_local - 1; ++i)
			for (int j = 1; j < n - 1; ++j)
				y_local[i * n + j] =
					(temp[(i + 1) * n + j] + temp[(i - 1) * n + j] +
					 temp[i * n + (j + 1)] + temp[i * n + (j - 1)] +
					 h * h * f((nums_local + i) * h, j * h)) *
					q;

		if (it % 2 == 0) {
			MPI_Waitall(2, send_req1, MPI_STATUSES_IGNORE);
			MPI_Waitall(2, recv_req1, MPI_STATUSES_IGNORE);
		} else {
			MPI_Waitall(2, send_req2, MPI_STATUSES_IGNORE);
			MPI_Waitall(2, recv_req2, MPI_STATUSES_IGNORE);
		}

		// MPI_Waitall(2, send_req1, MPI_STATUSES_IGNORE);
		// MPI_Waitall(2, recv_req1, MPI_STATUSES_IGNORE);

		// MPI_Barrier;

		/* пересчитываем верхние строки */
		if (id != 0)
			for (int j = 1; j < n - 1; ++j)
				y_local[j] = (temp[n + j] + y_prev_low[j] + temp[j + 1] +
							  temp[j - 1] + h * h * f(nums_local * h, j * h)) *
							 q;

		/* пересчитываем нижние строки */
		if (id != size - 1)
			for (int j = 1; j < n - 1; ++j)
				y_local[(str_local - 1) * n + j] =
					(y_next_top[j] + temp[(str_local - 2) * n + j] +
					 temp[(str_local - 1) * n + (j + 1)] +
					 temp[(str_local - 1) * n + (j - 1)] +
					 h * h * f((nums_local + (str_local - 1)) * h, j * h)) *
					q;

		norm_local = check_norm(temp, y_local);

		MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX,
					  MPI_COMM_WORLD);

		if (norm_err < eps) flag = false;
	}

	t1 += MPI_Wtime();

	for (int i = 0; i < size; ++i) {
		str_per_proc[i] *= n;
		nums_start[i] *= n;
	}

	MPI_Gatherv(y_local.data(), str_local * n, MPI_DOUBLE, y.data(),
				str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0,
				MPI_COMM_WORLD);

	if (id == 0) {
		// точное решение
		vector<double> check(n * n);
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j) check[i * n + j] = u(i * h, j * h);

		cout << "\n\n Якоби Isend + Irecv";
		cout << "\n\t norm: " << check_norm(y, check);
		cout << "\n\t iterations: " << it;
		printf("\n\t time: %.4f", t1);
	}

	delete[] send_req1;
	delete[] recv_req1;
	delete[] send_req2;
	delete[] recv_req2;
}

void Zeydel_ISend_IRecv(int id, int size) {
	vector<int> str_per_proc, nums_start;
	int str_local, nums_local;
	double norm_local, norm_err;

	MPI_Request* send_req1;
	MPI_Request* send_req2;
	MPI_Request* recv_req1;
	MPI_Request* recv_req2;

	int source_proc = id ? id - 1 : size - 1;
	int dest_proc = (id != (size - 1)) ? id + 1 : 0;

	int scount = (id != (size - 1)) ? n : 0;
	int rcount = id ? n : 0;

	str_split(id, size, str_local, nums_local, str_per_proc, nums_start);

	send_req1 = new MPI_Request[2], recv_req1 = new MPI_Request[2];
	send_req2 = new MPI_Request[2], recv_req2 = new MPI_Request[2];

	vector<double> y_local(str_local * n);
	vector<double> y_next_top(n);
	vector<double> y_prev_low(n);
	vector<double> temp(y_local.size());

	// пересылаем верхние и нижние строки temp
	MPI_Send_init(temp.data(), rcount, MPI_DOUBLE, source_proc, 0,
				  MPI_COMM_WORLD, send_req1);
	MPI_Recv_init(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 1,
				  MPI_COMM_WORLD, recv_req1);

	MPI_Send_init(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE,
				  dest_proc, 1, MPI_COMM_WORLD, send_req1 + 1);
	MPI_Recv_init(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 0,
				  MPI_COMM_WORLD, recv_req1 + 1);

	// пересылаем верхние и нижние строки y_local
	MPI_Send_init(y_local.data(), rcount, MPI_DOUBLE, source_proc, 0,
				  MPI_COMM_WORLD, send_req2);
	MPI_Recv_init(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 1,
				  MPI_COMM_WORLD, recv_req2);

	MPI_Send_init(y_local.data() + (str_local - 1) * n, scount, MPI_DOUBLE,
				  dest_proc, 1, MPI_COMM_WORLD, send_req2 + 1);
	MPI_Recv_init(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 0,
				  MPI_COMM_WORLD, recv_req2 + 1);

	vector<double> y;
	if (id == 0) y.resize(n * n);

	double t1 = -MPI_Wtime();

	int it = 0;
	bool flag = true;
	while (flag) {
		it++;
		// cout << "\n it = " << it;

		std::swap(temp, y_local);

		// y_local = temp;

		if (it % 2 == 0) {
			MPI_Startall(2, send_req1);
			MPI_Startall(2, recv_req1);
		} else {
			MPI_Startall(2, send_req2);
			MPI_Startall(2, recv_req2);
		}

		// внутренние строки (красные)
		for (int i = 1; i < str_local - 1; ++i)
			for (int j = ((i + 1) % 2 + 1); j < n - 1; j += 2)
				y_local[i * n + j] =
					(temp[(i + 1) * n + j] + temp[(i - 1) * n + j] +
					 temp[i * n + (j + 1)] + temp[i * n + (j - 1)] +
					 h * h * f((nums_local + i) * h, j * h)) *
					q;

		if (it % 2 == 0) {
			MPI_Waitall(2, send_req1, MPI_STATUSES_IGNORE);
			MPI_Waitall(2, recv_req1, MPI_STATUSES_IGNORE);
		} else {
			MPI_Waitall(2, send_req2, MPI_STATUSES_IGNORE);
			MPI_Waitall(2, recv_req2, MPI_STATUSES_IGNORE);
		}

		// верхние строки (красные)
		if (id != 0)
			for (int j = 2; j < n - 1; j += 2)
				y_local[j] = (temp[n + j] + y_prev_low[j] + temp[j + 1] +
							  temp[j - 1] + h * h * f(nums_local * h, j * h)) *
							 q;

		// нижние строки (красные)
		if (id != size - 1)
			for (int j = 1 + str_local % 2; j < n - 1; j += 2)
				y_local[(str_local - 1) * n + j] =
					(y_next_top[j] + temp[(str_local - 2) * n + j] +
					 temp[(str_local - 1) * n + (j + 1)] +
					 temp[(str_local - 1) * n + (j - 1)] +
					 h * h * f((nums_local + (str_local - 1)) * h, j * h)) *
					q;

		MPI_Barrier;

		if (it % 2 == 0) {
			MPI_Startall(2, send_req2);
			MPI_Startall(2, recv_req2);
		} else {
			MPI_Startall(2, send_req1);
			MPI_Startall(2, recv_req1);
		}

		// внутренние строки (чёрные)
		for (int i = 1; i < str_local - 1; ++i)
			for (int j = (i % 2 + 1); j < n - 1; j += 2)
				y_local[i * n + j] =
					(y_local[(i + 1) * n + j] + y_local[(i - 1) * n + j] +
					 y_local[i * n + (j + 1)] + y_local[i * n + (j - 1)] +
					 h * h * f((nums_local + i) * h, j * h)) *
					q;

		if (it % 2 == 0) {
			MPI_Waitall(2, send_req2, MPI_STATUSES_IGNORE);
			MPI_Waitall(2, recv_req2, MPI_STATUSES_IGNORE);
		} else {
			MPI_Waitall(2, send_req1, MPI_STATUSES_IGNORE);
			MPI_Waitall(2, recv_req1, MPI_STATUSES_IGNORE);
		}

		// верхние строки (чёрные)
		if (id != 0)
			for (int j = 1; j < n - 1; j += 2)
				y_local[j] =
					(y_local[n + j] + y_prev_low[j] + y_local[j + 1] +
					 y_local[j - 1] + h * h * f(nums_local * h, j * h)) *
					q;

		// нижние строки (чёрные)
		if (id != size - 1)
			for (int j = 1 + (str_local - 1) % 2; j < n - 1; j += 2)
				y_local[(str_local - 1) * n + j] =
					(y_next_top[j] + y_local[(str_local - 2) * n + j] +
					 y_local[(str_local - 1) * n + (j + 1)] +
					 y_local[(str_local - 1) * n + (j - 1)] +
					 h * h * f((nums_local + (str_local - 1)) * h, j * h)) *
					q;


		norm_local = check_norm(temp, y_local);

		MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX,
					  MPI_COMM_WORLD);

		if (norm_err < eps) flag = false;
	}

	t1 += MPI_Wtime();

	for (int i = 0; i < size; ++i) {
		str_per_proc[i] *= n;
		nums_start[i] *= n;
	}

	// MPI_Gather(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_local
	// * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gatherv(y_local.data(), str_local * n, MPI_DOUBLE, y.data(),
				str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0,
				MPI_COMM_WORLD);

	if (id == 0) {
		// точное решение
		vector<double> check(n * n);
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j) check[i * n + j] = u(i * h, j * h);

		cout << "\n\n Зейдель Isend + Irecv";
		cout << "\n\t norm: " << check_norm(y, check);
		cout << "\n\t iterations: " << it;
		printf("\n\t time: %.4f", t1);
	}

	delete[] send_req1;
	delete[] recv_req1;
	delete[] send_req2;
	delete[] recv_req2;
}
