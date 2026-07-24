import math

def get_alpha_beta(A, b):
    n = len(b)
    alpha = [[0.0] * n for _ in range(n)]
    beta = [0.0] * n
    for i in range(n):
        diag = A[i][i]
        beta[i] = b[i] / diag
        for j in range(n):
            if i != j:
                alpha[i][j] = -A[i][j] / diag
    return alpha, beta

def get_norm_matrix(matrix):
    return max(sum(abs(x) for x in row) for row in matrix)

def get_norm_vector(v):
    return max(abs(x) for x in v)

def simple_iteration(alpha, beta, eps):
    n = len(beta)
    x = list(beta)
    norm_a = get_norm_matrix(alpha)
    k = 0
    while True:
        k += 1
        x_new = [0.0] * n
        for i in range(n):
            s = sum(alpha[i][j] * x[j] for j in range(n))
            x_new[i] = beta[i] + s
        diff = [x_new[i] - x[i] for i in range(n)]
        if (norm_a / (1 - norm_a)) * get_norm_vector(diff) < eps:
            return x_new, k
        x = x_new

def seidel(alpha, beta, eps):
    n = len(beta)
    x = list(beta)
    norm_a = get_norm_matrix(alpha)
    k = 0
    while True:
        k += 1
        x_prev = list(x)
        for i in range(n):
            s = sum(alpha[i][j] * x[j] for j in range(n))
            x[i] = beta[i] + s
        diff = [x[i] - x_prev[i] for i in range(n)]
        if (norm_a / (1 - norm_a)) * get_norm_vector(diff) < eps:
            return x, k

A = [
    [-19, 2, -1, -8],
    [2, 14, 0, -4],
    [6, -5, -20, -6],
    [-6, 4, -2, 15]
]
b = [38, 20, 52, 43]
eps = 0.0001

alpha, beta = get_alpha_beta(A, b)
res_mpi, k_mpi = simple_iteration(alpha, beta, eps)
res_zei, k_zei = seidel(alpha, beta, eps)

print(f"MPI: {res_mpi}, Steps: {k_mpi}")
print(f"Seidel: {res_zei}, Steps: {k_zei}")