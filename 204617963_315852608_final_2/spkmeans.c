#define PY_SSIZE_T_CLEAN
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include "spkmeans.h"

void input_error(){
      printf("Invalid Input!");
      exit(1);
}
void print_elem(double e, int i, int N){
    (i!=N-1) ?
    (fabs(e)<0.0001) ? printf ("0.0000,") : printf("%.4f,",e)
    :
    (fabs(e)<0.0001) ? printf ("0.0000\n") : printf("%.4f\n", e);
}
/*if z==0 print A, else print diagonal of A*/
void print_resault(double **A, int N, int z){
    int i,j;
    for(i=0;i<N;i++){
        if(z)
            print_elem(A[i][i], i, N);
        else{
            for(j=0;j<N;j++)
                print_elem(A[i][j], j, N);
        }
    }
}
void free_matrix(double **matrix, int N){
    int i;
    for(i=0; i<N; i++)
        free(matrix[i]);
    free(matrix);
}
/* set C to A*B. | A,B,C:([N*N]) */
void matrix_mul(double **A, double **B, double **C, int N){
    double **D;
    int i,j,k;
    D = malloc_matrix(N,N);
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            D[i][j] = 0;
            for(k=0; k<N; k++)
                D[i][j]+= A[i][k]*B[k][j];
        }
    }
    for(i=0; i<N; i++)
        for(j=0; j<N; j++)
            C[i][j] = D[i][j];
    free_matrix(D,N);
}
/* set A to A transposed | A:([N*N]) */
void transpose(double** A, int N){
    int i,j;
    double tmp;
    for(i=0; i<N; i++){
        for(j=i+1; j<N; j++){
            tmp = A[i][j];
            A[i][j] = A[j][i];
            A[j][i] = tmp;
        }
    }
}
/* set IJ to be the indices of largest abs off-diagonal element 
of A, i.e A[IJ[0]][IJ[1]]=max{|A[i][j]|:i!=j}.  | A:([N*N]),IJ:([2]) */
void set_IJ(double **A, int *IJ, int N){
    double largest=-DBL_MAX;
    int i,j;
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            if(i!=j && fabs(A[i][j])>largest){
                largest = fabs(A[i][j]);
                IJ[0]=i; IJ[1]=j;  
            }
        }
    }
}
/* set P to rotation matrix */
void set_P(double **P, double c, double s, int I, int J, int N){
    int i,j;
    for(i=0; i<N; i++){        
        for(j=0; j<N; j++){
            if(i==j)
                P[i][j] = (i==I||i==J) ? c : 1 ;
            else if(i==J && j==I)
                P[i][j]=-s;
            else P[i][j] = (i==I && j==J) ? s : 0;
        }
    }
}
double** malloc_matrix(int N, int d){
    double** A;
    int i;
    A = malloc(N*sizeof(double*));
    for(i=0; i<N; i++)
        A[i] = malloc(d*sizeof(double));
    return A;
}
/* set I to id matrix | I:([N*N]) */
void ID(double **I, int N){
    int i,j;
    for(i=0;i<N;i++)
        for(j=0;j<N;j++)
            I[i][j] = i==j ? 1 : 0;
}
/* return off-diagonal sum of squares of A | A:([N*N]) */ 
double off(double** A, int N){
    double sum_square=0;
    int i,j;
    for(i=0; i<N; i++)
        for(j=0; j<N; j++)
            sum_square += i==j ? 0 : pow(A[i][j],2);
    return sum_square;
}
/* return Euclidean Norm of X-Y | X,Y:([d]) */
double euclidean_norm(double *X, double *Y, int d){
    double sum_square = 0;
    int i;
    for(i=0; i<d; i++)
        sum_square += pow(X[i]-Y[i],2);
    return sqrt(sum_square);
}
/* returns the index of the nearest centroid to a given vector  | V:([d]),C:([k*d]) */
int min_dist(double *V, double **C, int k, int d){
    double min = __INT_MAX__, dist;
    int i,  _index = -1; 
    for(i= 0;i<k; i++){
        dist = euclidean_norm(V,C[i],d);
        if(dist<min){
            min = dist; 
            _index = i;
        }       
    }
    return _index;
}
/* set W to weighted adjacency matrix of data A | A:([N*d]),W:([N*N]) */
void wam(double** A, double** W, int N, int d){
    int i,j;
    for(i=0; i<N; i++)
        for(j=0; j<N; j++)
            W[i][j] = i==j ? 0 : exp(euclidean_norm(A[i],A[j],d)/-2);    
}
/* set B to (diagonal degree matrix of A)^(-0.5)) | A,B:([N*N])  */
void ddg(double** A, double** B, int N){
    double sum;
    int i,j;
    for(i=0; i<N; i++){
        sum=0;
        for(j=0; j<N; j++){
            B[i][j] = 0;
            sum+=A[i][j];
        }
        B[i][i] = sum;
    }    
}
/* set W to normalized Laplacian matrix | W,D:([N*N]) */ 
void lnorm(double** W, double** D, int N){
    int i,j;
    for(i=0; i<N; i++)
        D[i][i]=1/sqrt(D[i][i]);
    matrix_mul(D,W,W,N); /* W = D*W */
    matrix_mul(W,D,W,N); /* W = D*W*D */
    for(i=0; i<N; i++)
        for(j=0; j<N; j++)
                W[i][j] = (i==j) ? 1-W[i][j] : -W[i][j];
}
/* set A to eiganvalues(on diagonal), and set V to eiganvectors(columns) | A,V:([N*N])  */ 
void jacobi(double** A, double** V, int N){
    double  **P, t;
    int *IJ, max_iter=100;
    P = malloc_matrix(N,N);   
    IJ = malloc(2*sizeof(int)); 
    while(max_iter>0){
        set_IJ(A, IJ, N); /* A[IJ[0]][IJ[1]] = largest off-diagonal element of A */
        t = (A[IJ[1]][IJ[1]] - A[IJ[0]][IJ[0]])/(2*A[IJ[0]][IJ[1]]); 
        if(t==0) t=1;
        else t=(t/fabs(t))/(fabs(t) + sqrt(1+t*t)); 
        set_P(P,1/sqrt(1+t*t),t/sqrt(1+t*t),IJ[0],IJ[1],N); /* P = rotation matrix */
        t = off(A, N); 
        transpose(P, N); /* P = Pt */
        matrix_mul(P, A, A, N); /* A = Pt*A */
        transpose(P, N); /* P = Pt (= original P) */
        matrix_mul(A, P, A, N); /* A = Pt*A*P */
        if(t - off(A, N) < pow(10,-5))/* convergence check */
             max_iter = 0;
        matrix_mul(V,P,V,N); /* V = V*P */
        max_iter--;
    }            
    free_matrix(P, N);
    free(IJ);
}
int main(int argc, char *argv[]){
    char *goal, *file_name, *delim =",", *ptr, line[1000];
    FILE *input_file = NULL;
    int N,d,i,j;
    double **df_c, **A, **B;
    if(argc != 3)
        input_error();
    goal = argv[1];
    file_name = argv[2];
    input_file = fopen(file_name, "r" );
    if(input_file==NULL)
        input_error();
    N = 0;
    d = 1;
    while(fgets(line, 1000, input_file)!=NULL) 
        N++; 
    input_file = fopen(file_name, "r" );
    fgets(line, 1000, input_file);
    for (i=0; (unsigned)i<strlen(line)-1; i++) 
        if(line[i]==',')
            d++;
    input_file = fopen(file_name, "r" );
    df_c = malloc(N*sizeof(double*));
    for(i=0; i<N; i++){
        df_c[i] = malloc(d*sizeof(double));
        fgets(line, 1000, input_file);
        ptr = strtok(line, delim);
        for(j=0; ptr!=NULL ;j++){
            df_c[i][j] = atof(ptr);
            ptr = strtok(NULL, delim);
        }
    }
    fclose(input_file);
    /*spkmeans */
    A = malloc_matrix(N,N);
    if(!strcmp(goal, "jacobi")){
        ID(A,N); /* init A to I */
        jacobi(df_c, A, N); /* df_c = eiganvalues, A = eiganvectors */
        print_resault(df_c, N, 1);
        print_resault(A, N, 0);  
    }
    else{
        wam(df_c, A, N, d);      /* A = weighted adjacency matrix of df_c */
        if(!strcmp(goal, "wam"))
            print_resault(A, N, 0);
        else{
            B = malloc_matrix(N,N);
            ddg(A, B, N);     /* B = diagonal degree matrix  */
            if(!strcmp(goal, "ddg"))
                print_resault(B, N, 0);
            else{
                lnorm(A, B, N); /* A = normalized Laplacian matrix */
                if (!strcmp(goal, "lnorm"))
                    print_resault(A, N, 0);
                else input_error();
                }
            free_matrix(B, N);
        }
    }
    free_matrix(A, N); 
    free_matrix(df_c, N);    
    return 0;
}


