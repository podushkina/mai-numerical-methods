import copy


def lu_decomposition(matrix, eps=1e-12):
    n = len(matrix)
    U = copy.deepcopy(matrix)
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    P = list(range(n))
    swaps = 0

    for k in range(n):
        max_row = k
        for i in range(k + 1, n):
            if abs(U[i][k]) > abs(U[max_row][k]):
                max_row = i

        if max_row != k:
            U[k], U[max_row] = U[max_row], U[k]
            P[k], P[max_row] = P[max_row], P[k]
            for j in range(k):
                L[k][j], L[max_row][j] = L[max_row][j], L[k][j]
            swaps += 1

        if abs(U[k][k]) < eps:
            raise ValueError("Матрица вырождена или близка к этому")

        for i in range(k + 1, n):
            mu = U[i][k] / U[k][k]
            L[i][k] = mu
            for j in range(k, n):
                U[i][j] -= mu * U[k][j]
            U[i][k] = 0

    sign = (-1) ** swaps
    return L, U, P, sign


def solve_system(L, U, P, b):
    n = len(L)
    z = [0.0] * n
    for i in range(n):
        sum_val = 0
        for j in range(i):
            sum_val += L[i][j] * z[j]
        z[i] = b[P[i]] - sum_val

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = 0
        for j in range(i + 1, n):
            sum_val += U[i][j] * x[j]
        x[i] = (z[i] - sum_val) / U[i][i]
    return x


def get_inverse(matrix):
    n = len(matrix)
    L, U, P, _ = lu_decomposition(matrix)
    inv = [[0.0] * n for _ in range(n)]

    for col in range(n):
        e = [1.0 if row == col else 0.0 for row in range(n)]
        res_col = solve_system(L, U, P, e)
        for row in range(n):
            inv[row][col] = res_col[row]
    return inv


A_17 = [
    [8, 8, -5, -8],
    [8, -5, 9, -8],
    [5, -4, -6, -2],
    [8, 3, 6, 6]
]
b_17 = [13, 38, 14, -95]

L, U, P, sign = lu_decomposition(A_17)
x_sol = solve_system(L, U, P, b_17)

det_A = sign
for i in range(len(U)):
    det_A *= U[i][i]

A_inv = get_inverse(A_17)

print("--- Решение СЛАУ ---")
for i, val in enumerate(x_sol):
    print(f"x{i + 1} = {val:.6f}")

print(f"\nОпределитель: {det_A:.4f}")

print("\n--- Обратная матрица ---")
for row in A_inv:
    print(" ".join([f"{val:10.6f}" for val in row]))