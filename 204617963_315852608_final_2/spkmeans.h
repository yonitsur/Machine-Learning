#ifndef SPKMEANS_H_
#define SPKMEANS_H_

void wam(double** A,double** B, int N, int d);
void ddg(double** A, double** B, int N);
void lnorm(double** W, double** D, int N);
void jacobi(double** A, double** V, int N);
void input_error();
void print_resault(double **A, int N, int z);
void free_matrix(double **matrix, int N);
void transpose(double** A, int N);
void matrix_mul(double **A, double **B, double **D, int N);
void set_IJ(double **A, int *IJ, int N);
void set_P(double **P, double c, double s, int I, int J, int N);
void ID(double **I, int N);
void print_elem(double e, int i, int N);
double** malloc_matrix(int N, int d);
double off(double **A, int N);
double euclidean_norm(double *X, double *Y, int d);
int min_dist(double *vector, double **centroids, int k, int d);
int main();

#endif