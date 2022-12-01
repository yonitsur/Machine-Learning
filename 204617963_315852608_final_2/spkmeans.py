import myspkmeanssp as spkmeans
import numpy as np
import pandas as pd
import sys
# eigenvalues are the first elemnet in each row of U. 
# eigenvectors are the rows of U from index 1. 
# sotrt rows of U in increasing order of eigenvalues.
def quicksort(U, l, r):
   if(l<r):
      pivot,i,j = l,l,r
      while(i<j):
         while(U[i][0]<=U[pivot][0] and i<r):
            i+=1
         while(U[j][0]>U[pivot][0]):
            j-=1
         if(i<j):
            temp=U[i]
            U[i]=U[j]
            U[j]=temp 
      temp=U[pivot]
      U[pivot]=U[j]
      U[j]=temp
      quicksort(U,l,j-1)
      quicksort(U,j+1,r) 
       
def print_indices(ind):
    for i in range(len(ind)):
        ind[i] = (str)(ind[i])
    print(','.join(ind))  

def print_results(arr):
    for row in arr:
        for i in range(len(row)):
            row[i] = format(row[i],".4f")
            if row[i]=="-0.0000":
                row[i] = "0.0000"
        print(','.join(row))  

def eigengap_heuristic(U):
    A = np.array([])
    for i in range(len(U)//2):
        A = np.append(A, abs(U[0][i]-U[0][i+1]))
    return np.argmax(A)

def kmeans_pp(df, N, d, k):
    np.random.seed(0)
    index = np.random.choice(N) # generate rand index_0 . µ0 = df[index_0]  
    C = np.append(np.empty((0,d), float), np.array([df[index]]), axis=0) # C[0]=µ0
    C_indices = [index] # C_indices[0]= index_0
    i=1
    while(i<k):
        D = np.zeros(N) # D[l] = min{(xl − µj)^2}, ∀j 1≤j≤i
        P = np.zeros(N) # P[l] = D[l]/sum(D)
        for l in range(N): 
            D[l]= np.min([np.linalg.norm(df[l]-C[j])**2 for j in range(i)])
        for l in range(N): 
            P[l] = D[l]/np.sum(D)
        index = np.random.choice(N, p=P) # generate rand index_i according to P
        C_indices.append(index) # C_indices[i]= index_i
        C = np.append(C, np.array([df[index]]), axis=0) # C[i]=µi=df[index_i], ∀i 0≤i<k 
        i+=1  
    try:
        final_centroids = spkmeans.fit(df, C.tolist(), d, -1) 
    except:
        raise Exception("An Error Has Occurred")
    print_indices(C_indices)
    print_results(final_centroids)

# spkmeans #
try:
    k = int(sys.argv[1]) # number of clusters
    goal = sys.argv[2]
    file_name = sys.argv[3] 
except:
    raise Exception("Invalid Input!")

dict = {"wam":1, "ddg":2, "lnorm":3, "spk":-2, "jacobi":0} # "T":4, "kmeans":-1

if goal not in dict.keys():
    raise Exception("Invalid Input!")

try:
    df = pd.read_csv(file_name, header=None)
except:
    raise Exception("Invalid Input!") 

N = df.shape[0] 
d = df.shape[1] 
df = df.to_numpy().tolist() 

if k>=N or k<0:
    raise Exception("Invalid Input!")  
try:
    final_matrix = spkmeans.fit(df, None, d, dict[goal]) # exe spkmeans on the data
except:
    raise Exception("An Error Has Occurred")
        
if(goal=="spk"):
    if(k==0):
        k = eigengap_heuristic(final_matrix)
    final_matrix = np.transpose(np.array(final_matrix)).tolist()
    quicksort(final_matrix, 0, len(final_matrix)-1)
    final_matrix = np.transpose(np.array(final_matrix))
    T = spkmeans.fit(final_matrix[1:,:k].tolist(), None, k, 4)
    kmeans_pp(T, N, k, k)
else:
    print_results(final_matrix)
  
  



    
