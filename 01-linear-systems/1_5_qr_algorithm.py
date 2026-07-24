import math


def matrix_multiply(A, B):
    n, m, k = len(A), len(A[0]), len(B[0])
    C = [[0.0] * k for _ in range(n)]
    for i in range(n):
        for j in range(k):
            for l in range(m):
                C[i][j] += A[i][l] * B[l][j]
    return C


def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def get_norm(v):
    return math.sqrt(sum(x ** 2 for x in v))


def qr_decomposition(A):
    n = len(A)
    R = [row[:] for row in A]
    Q = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for k in range(n - 1):
        # Выделяем столбец для преобразования Хаусхолдера
        b = [R[i][k] for i in range(k, n)]
        norm_b = get_norm(b)

        v = [0.0] * len(b)
        # Формула вектора отражения v
        sign = 1.0 if b[0] >= 0 else -1.0
        v[0] = b[0] + sign * norm_b
        for i in range(1, len(b)):
            v[i] = b[i]

        norm_v = get_norm(v)
        if norm_v == 0: continue

        # Строим матрицу Хаусхолдера H = E - 2vv^T / v^Tv
        H_small = [[1.0 if i == j else 0.0 for j in range(len(b))] for i in range(len(b))]
        for i in range(len(b)):
            for j in range(len(b)):
                H_small[i][j] -= 2.0 * v[i] * v[j] / (norm_v ** 2)

        H = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for i in range(len(b)):
            for j in range(len(b)):
                H[i + k][j + k] = H_small[i][j]

        R = matrix_multiply(H, R)
        Q = matrix_multiply(Q, H)

    return Q, R


def extract_eigenvalues(A, eps):
    n = len(A)
    eigenvalues = []
    i = 0
    while i < n:
        # Проверка на наличие поддиагонального элемента (комплексная пара)
        lower_sum = math.sqrt(sum(A[row][i] ** 2 for row in range(i + 1, n)))

        if i == n - 1 or lower_sum < eps:
            eigenvalues.append(round(A[i][i], 4))
            i += 1
        else:
            # Решаем квадратное уравнение для блока 2x2
            a11, a12 = A[i][i], A[i][i + 1]
            a21, a22 = A[i + 1][i], A[i + 1][i + 1]

            # Характеристическое уравнение: L^2 - (a11+a22)L + (a11*a22 - a12*a21) = 0
            trace = a11 + a22
            det = a11 * a22 - a12 * a21
            discriminant = trace ** 2 - 4 * det

            if discriminant >= 0:
                eigenvalues.append(round((trace + math.sqrt(discriminant)) / 2, 4))
                eigenvalues.append(round((trace - math.sqrt(discriminant)) / 2, 4))
            else:
                real = round(trace / 2, 4)
                imag = round(math.sqrt(-discriminant) / 2, 4)
                eigenvalues.append(complex(real, imag))
                eigenvalues.append(complex(real, -imag))
            i += 2
    return eigenvalues


def qr_algorithm(A, eps):
    A_k = [row[:] for row in A]
    prev_vals = []

    for iteration in range(1, 1000):
        Q, R = qr_decomposition(A_k)
        # Перемножаем в обратном порядке: A_next = R * Q
        A_k = matrix_multiply(R, Q)

        curr_vals = extract_eigenvalues(A_k, eps)

        if prev_vals:
            # Условие остановки по разности собственных значений
            diff = max(abs(c - p) for c, p in zip(curr_vals, prev_vals))
            if diff < eps:
                print(f"Сошлось за {iteration} итераций")
                return curr_vals
        prev_vals = curr_vals
    return prev_vals


# ==========================================
# БЛОК УЛУЧШЕННОГО ВЫВОДА ДЛЯ ОТЧЕТА
# ==========================================

# Твоя исходная матрица (Вариант 17)
matrix_17 = [
    [-6, 1, -4],
    [-6, 8, -2],
    [2, -9, 5]
]

print("1. Исходная матрица A:")
for row in matrix_17:
    print("  [" + " ".join([f"{val:8.4f}" for val in row]) + " ]")
print("-" * 50)

# Получаем QR-разложение для исходной матрицы
Q, R = qr_decomposition(matrix_17)

print("2. QR-разложение матрицы A:")
print("Матрица Q (ортогональная):")
for row in Q:
    print("  [" + " ".join([f"{val:8.4f}" for val in row]) + " ]")

print("\nМатрица R (верхняя треугольная):")
for row in R:
    print("  [" + " ".join([f"{val:8.4f}" for val in row]) + " ]")
print("-" * 50)

# Проверка: перемножаем Q * R обратно
A_reconstructed = matrix_multiply(Q, R)
print("3. Проверка разложения (Q * R):")
for row in A_reconstructed:
    print("  [" + " ".join([f"{val:8.4f}" for val in row]) + " ]")
print("-" * 50)

# Запуск самого QR-алгоритма для поиска собственных значений
eps_value = 0.001
result = qr_algorithm(matrix_17, eps_value)

print(f"4. QR-алгоритм для собственных значений (eps = {eps_value}):")
print("Собственные значения lambda_i:")
for i, val in enumerate(result):
    if isinstance(val, complex):
        # Красивый вывод комплексных чисел, если они есть
        print(f"  lambda_{i + 1} = {val.real:.4f} {'+' if val.imag >= 0 else '-'} {abs(val.imag):.4f}i")
    else:
        print(f"  lambda_{i+1} = {val:.4f}")