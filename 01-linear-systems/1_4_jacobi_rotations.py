import math


def get_max_off_diagonal(A):
    n = len(A)
    max_val = 0
    p, q = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                p, q = i, j
    return p, q, max_val


def matrix_multiply(A, B):
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def jacobi_rotations(matrix, eps):
    n = len(matrix)
    A = [row[:] for row in matrix]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    while True:
        p, q, max_val = get_max_off_diagonal(A)

        off_diag_norm = 0
        for i in range(n):
            for j in range(i + 1, n):
                off_diag_norm += A[i][j] ** 2
        off_diag_norm = math.sqrt(2 * off_diag_norm)

        if off_diag_norm < eps:
            break

        if abs(A[p][p] - A[q][q]) < 1e-10:
            phi = math.pi / 4
        else:
            phi = 0.5 * math.atan(2 * A[p][q] / (A[p][p] - A[q][q]))

        c, s = math.cos(phi), math.sin(phi)

        U = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        U[p][p], U[q][q] = c, c
        U[p][q], U[q][p] = -s, s

        A = matrix_multiply(transpose(U), matrix_multiply(A, U))
        V = matrix_multiply(V, U)

    eigenvalues = [A[i][i] for i in range(n)]
    eigenvectors = transpose(V)
    return eigenvalues, eigenvectors


A_17 = [
    [5, -3, -4],
    [-3, -3, 4],
    [-4, 4, 0]
]

vals, vecs = jacobi_rotations(A_17, 0.0001)

print("Eigenvalues:")
print([round(v, 4) for v in vals])
print("Eigenvectors:")
for vec in vecs:
    print([round(v, 4) for v in vec])