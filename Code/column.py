import numpy as np

def Colonne(A,j):
    A = np.transpose(A)
    return np.transpose(A[j])

# A=np.ones(12)
# print(A)
# B=np.transpose(A)
# print(B)

# print(np.shape(A))
