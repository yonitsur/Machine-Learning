#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h> 
#include <string.h>
#include "spkmeans.h"

/*if (z==0) set final_matrix to A. else (z==1) set final_matrix 
to B + new first raw with the diagonal of A, i.e final_matrix:([N+1*k]) | A,B:([N*k]) , z:0/1 */
void set_final_martix(PyObject* final_matrix, double **A, double **B, int N, int k, int z){
    int i,j;
    PyObject* rows;
    for(i=0; i<N+z; i++){
        rows = PyList_New(k);
        for(j=0; j<k; j++){
            if(z)
                (i==0) ? PyList_SetItem(rows, j, Py_BuildValue("d", fabs(A[j][j])))
                       : PyList_SetItem(rows, j, Py_BuildValue("d", B[i-1][j]));
            else PyList_SetItem(rows, j, Py_BuildValue("d", A[i][j]));
        }
        PyList_SetItem(final_matrix, i, Py_BuildValue("O", rows));
    }
}
double** set_data(PyObject* py_ob, int N, int k){
    double **R;
    int i,j;
    R = malloc(N*sizeof(double*));
    for(i=0; i<N; i++){
        R[i] = malloc(k*sizeof(double));
        for(j=0; j<k; j++) 
            R[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(py_ob, i), j));
    }
    return R;
}
double calc_T(double** R, int k, int i, int j){
    double sum_sq = 0;
    int l;
    for (l=0; l<k; l++)
        sum_sq+=R[i][l]*R[i][l];
    sum_sq = sqrt(sum_sq);
    return (sum_sq==0) ? 0 : R[i][j]/sum_sq;
}
PyObject* setT(PyObject* U, int k){
    double **T, **R;
    int N,i,j;
    N = PyList_Size(U);
    R = set_data(U, N, k);
    T = malloc_matrix(N,k);
    for(i=0; i<N; i++)
        for (j=0; j<k; j++)
            T[i][j] = calc_T(R, k, i, j); 
    U =  PyList_New(N);
    set_final_martix(U, T, NULL ,N, k, 0); /* U=T */
    free_matrix(R, N);
    free_matrix(T, N);
    return U;
}
PyObject* kmeans(PyObject* vectors_py, PyObject* centroids_py, int d){
    int N, k, i, j, nearest_cent, max_iter=300, *cluster_size;
    double **centroids, **vectors, **weights, **cluster_weight;
    N = PyList_Size(vectors_py);
    k = PyList_Size(centroids_py);
    vectors = set_data(vectors_py, N, d);
    centroids = set_data(centroids_py, k, d);
    weights = malloc_matrix(k,d);
    cluster_weight = malloc_matrix(k,d);
    cluster_size =malloc(k*sizeof(int)); 
    while(max_iter>0){
        for(i=0; i<k; i++){ 
            cluster_size[i]=0;
            for(j=0; j<d; j++)
                cluster_weight[i][j] = 0;
        }
        for(i=0; i<N; i++){
            nearest_cent = min_dist(vectors[i], centroids, k, d);
            cluster_size[nearest_cent]++; 
            for(j=0; j<d;j++)
                cluster_weight[nearest_cent][j] += vectors[i][j]; 
        }
        for(i=0; i<k; i++){
            for(j=0; j<d; j++)
                weights[i][j] = (cluster_size[i]==0) ? 0 : cluster_weight[i][j]/cluster_size[i]; 
            if(euclidean_norm(centroids[i], weights[i], d) < 0) 
                max_iter=0;
            for(j=0; j<d; j++)
                centroids[i][j]=weights[i][j];                  
        }
        max_iter--;
    } 
    vectors_py =  PyList_New(k);
    set_final_martix(vectors_py, centroids, NULL ,k, d, 0); 
    free_matrix(centroids, k);
    free_matrix(vectors, N);
    free_matrix(weights, k);
    free_matrix(cluster_weight, k);
    free(cluster_size);
    return vectors_py;
}
/* goal:    0 : jacobi
            1 : wam
            2 : ddg
            3 : lnorm 
            4 : calculate T
           -1 : kmeans 
           -2 : spk */
static PyObject* fit(PyObject* self, PyObject* args){
    PyObject *df_py, *centroids_py;
    int d, N, goal;
    double **df_c, **A, **B;
    PyArg_ParseTuple(args, "OOii", &df_py, &centroids_py, &d, &goal);
    if(goal==-1) /* kmeans */
        return kmeans(df_py, centroids_py, d);
    if (goal == 4) 
        return setT(df_py, d);
    N = PyList_Size(df_py);
    df_c = set_data(df_py, N, d);
    df_py = PyList_New(N+(goal>0?0:1)); /* for spk/jacobi we need N+1 rows, else N. */
    A = malloc_matrix(N,N); 
    if(goal==0){ /*jaacobi*/ 
        ID(A, N);
        jacobi(df_c, A, N); 
        set_final_martix(df_py ,df_c, A, N, N, 1);
    } 
    else{  
        wam(df_c, A, N, d); /* A = weighted adjacency matrix */
        if(goal==1) /* wam */
            set_final_martix(df_py, A, NULL, N, N, 0);
        else{
            B = malloc_matrix(N,N);
            ddg(A, B, N); /* B = (diagonal degree matrix)^-0.5 */
            if(goal==2) /* ddg */
                set_final_martix(df_py, B, NULL, N, N, 0); 
            else{
                lnorm(A, B, N); /* A = normalized Laplacian matrix */
                if(goal==3) /* lnorm */
                    set_final_martix(df_py, A, NULL, N, N , 0);
                else{ /* spk */
                    ID(B, N); /* set B to ID matrix*/
                    jacobi(A, B, N); /* A = eiganvalues, B = eiganvectors */
                    set_final_martix(df_py ,A, B, N, N, 1);
                }
            }
            free_matrix(B, N);
        } 
    }
    free_matrix(A, N);
    free_matrix(df_c, N);
    return df_py;
}
/* C-Python API */
static PyMethodDef spkmeansMethods[] = {
        {"fit", (PyCFunction) fit, METH_VARARGS, PyDoc_STR("C spkmeans algorithm")},
        {NULL, NULL, 0, NULL}
};
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "myspkmeanssp", NULL, -1, spkmeansMethods
};
PyMODINIT_FUNC
PyInit_myspkmeanssp(void){
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}



