// Direct problem

#if 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include <omp.h>

#define M 3 
#define N 2
#define T 10
#define t_mesh 100
#define N_t (T)*(t_mesh)

#define eps 0.00000000001

#define N_meas 5

#define N_prm ((N)+(M))

#define pi 3.1415926535897932384626433832795

void clear_arr_sm(int n, double* arr);
void clear_arr_bg(int n, double* arr);

void B(double t, double *x, double *prm/*input*/, double* f/*output*/)
{
	clear_arr_sm(N, f);

	f[0] = x[0] * (prm[0] - prm[1] * x[1]);
	f[1] = prm[2] * x[0] * x[1];

	return;
}

double Runge_Kutta(double *y0, double* y_k, double *prm, double *k1, double *k2, double *k3, double *k4, double **x)
{
	int i, j, k;
	double h_t;
	double t_k;

	int n = N;
	int nt1 = N_t + 1;
	int ione = 1;

	double *help;
	help = new double[n];

	double *f;		//right hand side
	f = new double[n];

	h_t = double(T) / double(N_t);


	for(int i = 0; i < n; i++)
#pragma omp parallel for simd schedule(simd:static)
		for (int j = 0; j < nt1; j++)
			x[i][j] = 0;

	// initial conditions

#pragma omp for simd
	for (int i = 0; i < n; i++)
	{
		x[i][0] = y0[i];
		y_k[i] = y0[i];
	}

	for (k = 0; k < N_t; k++)     // count is until the last point of T
	{
		t_k = k * h_t;

		B(t_k, y_k, prm, f); // massive of right parts transfer from function correctly
		LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', 1, n, f, n, k1, n);

#pragma omp for simd
		for(int i = 0; i < n; i++)
			help[i] = y_k[i] + h_t * 0.5 * k1[i];

		B(t_k + h_t * 0.5, help, prm, f);
		LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', 1, n, f, n, k2, n);


#pragma omp for simd
		for (int i = 0; i < n; i++)
			help[i] = y_k[i] + h_t * 0.5 * k2[i];

		B(t_k + h_t * 0.5, help, prm, f);
		LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', 1, n, f, n, k3, n);
	
#pragma omp for simd
		for (int i = 0; i < n; i++)
			help[i] = y_k[i] + h_t * k3[i];

		B(t_k + h_t, help, prm, f);
		LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', 1, n, f, n, k4, n);

#pragma omp for simd
		for (int i = 0; i < N; i++)
		{
			x[i][k + 1] = x[i][k] + h_t*(k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / double(6.0);
			y_k[i] = x[i][k + 1];// for next step
		}

	}

	delete[] f;
	delete[] help;

	return 0;
}

double lin_inter(double x0, double fx0, double x1, double fx1, double x)
{
	return (x - x0) / (x1 - x0) * (fx1 - fx0) + fx0;
}

void clear_arr_sm(int n, double* arr)
{
#pragma omp parallel for simd schedule(simd:static)
	for (int i = 0; i < n; i++)
		arr[i] = 0;
}

void clear_arr_bg(int n, double* arr)
{
#pragma omp for simd schedule(simd:static)
	for (int i = 0; i < n; i++)
		arr[i] = 0;
}

int main()
{

	int i, j, k, l;

	double h_t = double(T) / double(N_t);

	int size = N * (N_t + 1);
	int n = N;
	int m = M;
	int nprm = N_prm;
	int nt1 = N_t + 1;

	double *k1, *k2, *k3, *k4; // vectors of Runge-Kutta method

	double *y; // vector of solution
	double *y_init; // initial condition

	int h_meas = 0;
	double alpha = 0;

	y = new double[n];
	y_init = new double[n];

	k1 = new double[n];
	k2 = new double[n];
	k3 = new double[n];
	k4 = new double[n];

	clear_arr_sm(n, k1);
	clear_arr_sm(n, k2);
	clear_arr_sm(n, k3);
	clear_arr_sm(n, k4);

	double** X_exact; /* matrix of solution: y0 - t0, t1, t2, ... ,tN
				y1 - t0, t1, t2, ..., tN
				y2 - t0, t1, t2, ... ,tN
				direct problem								*/

	X_exact = new double*[n];
	for (int i = 0; i < n; i++)
		X_exact[i] = new double[nt1]; // since 0 to N_t

	const double b11 = 0.7, b12 = 0.6, b21 = 0.4, X10 = 1.6, X20 = 1.7; //unknown parameters

	clear_arr_sm(n, y);
	y_init[0] = X10;
	y_init[1] = X20;

	double *prm_app, *prm_ex;
	prm_ex = new double[nprm];
	prm_app = new double[nprm];

	prm_ex[0] = b11; prm_ex[1] = b12; prm_ex[2] = b21; prm_ex[3] = X10; prm_ex[4] = X20;

	/**************************************/

	printf("Direct problem solution of exact data...\n");

	Runge_Kutta(y_init, y, prm_ex, k1, k2, k3, k4, X_exact);

	/***********Printing***************/
	FILE *out, *out1;
	double t_k;
	out = fopen("DP_output_before.dat", "w");

	for (int k = 0; k < nt1; k++)
	{
		t_k = k*h_t;
		fprintf(out, "%5.4lf %lf %lf \n", t_k, X_exact[0][k], X_exact[1][k]);

	}
	fclose(out);


	/**************************************/
	printf("\nSizes of matrices:\n");
	printf("A: %d x %d\nq: %d x %d\nf: %d x %d\n", size, nprm, nprm, 1, size, 1);

	printf("\nMesh:\n%4d\n", int(N_t));

	printf("\nMeasurements:\n%d\n", int(N_meas));

	system("pause");

	// for discretization

	double* A;
	A = new double[size * nprm];

	int lda = nprm; // a[i][j]=a[i*lda+j]
	int ldm = n;
	int ldp = m;

	double *M_i, *P_i;
	M_i = new double[n * ldm];
	P_i = new double[n * ldp];

	double *M_res, *P_res;
	M_res = new double[n * ldm];
	P_res = new double[n * ldp];

	double *M_help;
	M_help = new double[n * ldm];

	// for SVD

	double *sing, *u, *vt, *superb;

	int ldu = size;
	int ldvt = nprm;

	sing = new double[nprm];
	u = new double[size * ldu];
	vt = new double[nprm * ldvt];
	superb = new double[nprm];

	// for Direct Problem solutions

	double** X_app, **X_meas, **X_app_meas;
	X_app = new double*[n];
	X_meas = new double*[n];
	X_app_meas = new double*[n];

	for (int i = 0; i < n; i++)
	{
		X_app[i] = new double[nt1]; // since 0 to N_t
		X_meas[i] = new double[nt1];
		X_app_meas[i] = new double[nt1]; // since 0 to N_t
	}

	// for linear interpolation and SVD
	double *f, *g, *q, *z;
	q = new double[nprm];
	z = new double[nprm];
	f = new double[size];
	g = new double[size];

	clear_arr_sm(nprm, q);
	clear_arr_sm(nprm, z);
	clear_arr_bg(size, f);
	clear_arr_bg(size, g);


	/*********ITERATION STEPS********/
	double norm1 = 0, norm2 = 0;
	int iter = 0;

	double timer1, timer2;
	
	timer1 = omp_get_wtime();

	do
	{
		iter++;
		printf("\n****************\nIteration: %d\n", iter);
		if (iter == 1)
		{
			const double b11_app = 0.5, b12_app = 0.5, b21_app = 0.6, X10_app = 1.8, X20_app = 1.8;
			prm_app[0] = b11_app; prm_app[1] = b12_app; prm_app[2] = b21_app; prm_app[3] = X10_app; prm_app[4] = X20_app; // intitial approximations
		}

		clear_arr_sm(n, y);
		y_init[0] = prm_app[3];
		y_init[1] = prm_app[4];

		printf("Direct problem solution of approximate data...\n");

		Runge_Kutta(y_init, y, prm_app, k1, k2, k3, k4, X_app);

		/***********Printing***************/
		if (iter == 1)
		{
			out = fopen("DP_app_output_before_meas_5.dat", "w");

			for (int k = 0; k < N_t + 1; k++)
			{
				t_k = k*h_t;
				fprintf(out, "%5.4lf %lf %lf \n", t_k, X_app[0][k], X_app[1][k]);

			}
			fclose(out);
		}


		/*************************************************/

		/*********Linear interpolation of measurements****/

		for (int i = 0; i < n; i++)
#pragma omp parallel for simd schedule(simd:static)
			for (int j = 0; j < nt1; j++)
			{
				X_meas[i][j] = 0;
				X_app_meas[i][j] = 0;
			}

		h_meas = int(N_t) / int(N_meas);
		printf("h_meas: %d\n", h_meas);

		// filling the measurements
		for (int j = 0; j < N_meas + 1; j++)
		{
			X_meas[0][j*h_meas] = X_exact[0][j*h_meas];
			X_meas[1][j*h_meas] = X_exact[1][j*h_meas];

			X_app_meas[0][j*h_meas] = X_app[0][j*h_meas];
			X_app_meas[1][j*h_meas] = X_app[1][j*h_meas];
		}

		// linear interpolation of other point in the grid
		for (int j = 0; j < N_meas; j++)
		for (int i = 0; i < h_meas; i++)
		{
			X_meas[0][i + j*h_meas] = lin_inter(j*h_meas, X_exact[0][j*h_meas], (j + 1)*h_meas, X_exact[0][(j + 1)*h_meas], i + j*h_meas);
			X_meas[1][i + j*h_meas] = lin_inter(j*h_meas, X_exact[1][j*h_meas], (j + 1)*h_meas, X_exact[1][(j + 1)*h_meas], i + j*h_meas);

			X_app_meas[0][i + j*h_meas] = lin_inter(j*h_meas, X_app[0][j*h_meas], (j + 1)*h_meas, X_app[0][(j + 1)*h_meas], i + j*h_meas);
			X_app_meas[1][i + j*h_meas] = lin_inter(j*h_meas, X_app[1][j*h_meas], (j + 1)*h_meas, X_app[1][(j + 1)*h_meas], i + j*h_meas);
		}


		/*********Construting of matrix A********/

		clear_arr_bg(size * nprm, A);
		clear_arr_sm(n * ldm, M_res);
		clear_arr_sm(n * ldm, M_i);
		clear_arr_sm(n * ldp, P_res);
		clear_arr_sm(n * ldp, P_i);
		clear_arr_sm(n * ldm, M_help);


		for (int i = 0; i < n; i++)
			M_res[i*ldm + i] = 1.0;

		alpha = 0;

		for (int j = 0; j < nt1; j++)
		{

			M_i[0 * ldm + 0] = 1 + h_t*(prm_app[0] - prm_app[1] * X_app[1][j]);
			M_i[0 * ldm + 1] = -h_t*prm_app[1] * X_app[0][j];
			M_i[1 * ldm + 0] = h_t*prm_app[2] * X_app[1][j];
			M_i[1 * ldm + 1] = 1 + h_t*prm_app[2] * X_app[0][j];

			P_i[0 * ldp + 0] = h_t*X_app[0][j];
			P_i[0 * ldp + 1] = -h_t*X_app[0][j] * X_app[1][j];
			P_i[0 * ldp + 2] = 0;

			P_i[1 * ldp + 0] = 0;
			P_i[1 * ldp + 1] = 0;
			P_i[1 * ldp + 2] = h_t*X_app[0][j] * X_app[1][j];


			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, M_i, ldm, M_res, ldm, 0.0, M_help, ldm);
			LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', n, n, M_help, ldm, M_res, ldm);

			if (j == 0) alpha = 0.0;
			else alpha = 1.0;

			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, m, n, alpha, M_i, ldm, P_res, ldp, 1.0, P_i, ldp);
			LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', n, m, P_i, ldp, P_res, ldp);

			// -------------------Fulfilling matrix A-----------------

			LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', n, m, P_res, ldp, &A[j* n * lda], lda);
			LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', n, n, M_res, ldm, &A[j* n * lda + m], lda);
		}

		// some output
		if (iter == 1)
		{
			out1 = fopen("A_202_5.dat", "w");
			for (int i = 0; i < size; i++)
			{
				for (int j = 0; j < 5; j++)
				{
				//	printf("%12.9lf ", A[i*lda + j]);
					fprintf(out1, "%12.9lf ", A[i * lda + j]);
				}
				fprintf(out1, "\n");
				//printf("\n");
			}

			fclose(out1);
		}
        
		
		// SVD decomposition
		LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', size, nprm, A, lda, sing, u, ldu, vt, ldvt, superb);

		// singular values in array SING; arrays U and VT - ortogonal matrices from U * SIGMA * VT * q = f
		printf("\nSingular values:\n");
		for (int i = 0; i < nprm; i++)
			printf("%5.4lf\n", sing[i]);

		// fulfilling of right part F
#pragma omp parallel for simd schedule(simd:static)
		for (int i = 0; i < size; i++)
		{
			int i2 = i / 2;
			//	if (i%2==0) f[i] = X_exact[0][i/2] - X_app[0][i/2];
			//	else f[i] = X_exact[1][int(i/2.0-0.5)] - X_app[1][int(i/2.0-0.5)];

			if (i % 2 == 0) f[i] = X_meas[0][i2] - X_app_meas[0][i2];
			else f[i] = X_meas[1][int(i2 - 0.5)] - X_app_meas[1][int(i2 - 0.5)];
		}

		// some output
		if (iter == 1)
		{
			out1 = fopen("N1_N2_interpolation_size_101_meas_5.dat", "w");
			for (int i = 0; i < nt1; i++)
			{
				fprintf(out1, "%12.9lf %12.9lf %12.9lf %12.9lf %12.9lf %12.9lf\n", X_meas[0][i], X_meas[1][i], X_app_meas[0][i], X_app_meas[1][i], X_meas[0][i]-X_app_meas[0][i],X_meas[1][i]-X_app_meas[1][i]);
			}

			fclose(out1);
		}

		// some output
		if (iter == 1)
		{
			out1 = fopen("u_v_Dasha.dat", "w");
			fprintf(out1, "-------u-------\n");
			for (int i = 0; i < size; i++)
			{
				fprintf(out1, "%d  ", i);
				for (int j = 0; j < size; j++)
					fprintf(out1, "%7.4lf ", u[i*ldu + j]);
				fprintf(out1,"\n");
			}
			fprintf(out1, "\n-------vt-------\n");
			for (int i = 0; i < nprm; i++)
			{
				fprintf(out1, "%d  ", i);
				for (int j = 0; j < nprm; j++)
					fprintf(out1, "%7.4lf ", vt[i*ldvt + j]);
				fprintf(out1, "\n");
			}
			fclose(out1);
		}


		// computing g = UT * f
		cblas_dgemv(CblasRowMajor, CblasTrans, size, size, 1.0, u, ldu, f, 1, 0.0, g, 1);

		// computing z[i] = g[i] / sing [i]
		for (int i = 0; i < nprm; i++)
			z[i] = g[i] / sing[i];

		// computing q = V * z
		cblas_dgemv(CblasRowMajor, CblasTrans, nprm, nprm, 1.0, vt, ldvt, z, 1, 0.0, q, 1);

		// result is in vector q
		if (iter == 1)
		{
			printf("\nUnknown delta-vector of parameters q:\n");
			for (int i = 0; i < nprm; i++)
			printf("%5.4lf\n", q[i]);
		}

		printf("\nUnknown vector of parameters q:\n");
		for (int i = 0; i < nprm; i++)
			printf("gained: %5.4lf exact: %5.4lf\n", q[i] + prm_app[i], prm_ex[i]);

		norm1 = 0, norm2 = 0;

		for (int i = 0; i < nprm; i++)
		{
			prm_app[i] = q[i] + prm_app[i];
			norm1 += (prm_app[i] - prm_ex[i])*(prm_app[i] - prm_ex[i]);
			norm2 += prm_ex[i] * prm_ex[i];
		}
			
		norm1 = sqrt(norm1) / sqrt(norm2);
		printf("norma %lf\n", norm1);
		//if (iter == 1) system("pause");

	} while (norm1 > eps);
	
	timer2 = omp_get_wtime() - timer1;

	printf("Time: %lf\n", timer2);



	delete[] k1;
	delete[] k2;
	delete[] k3;
	delete[] k4;

	delete[] y;
	delete[] y_init;

	delete[] A;

	delete[] M_i;
	delete[] P_i;

	delete[] M_res;
	delete[] P_res;
	delete[] M_help;

	delete[] sing;
	delete[] u;
	delete[] vt;
	delete[] superb;

	delete[] f;
	delete[] g;
	delete[] q;
	delete[] z;

	for (int i = 0; i < n; i++)
	{
		delete[] X_app[i]; 
		delete[] X_meas[i];
		delete[] X_app_meas[i];
		delete[] X_exact[i];
	}

	delete[] X_app;
	delete[] X_meas;
	delete[] X_app_meas;
	delete[] X_exact;



	system("pause");
	return 0;


}

#endif