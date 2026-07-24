def solve_progonka(a, b, c, d):
    n = len(d)

    # проверка устойчивости: |bi| >= |ai| + |ci|
    for i in range(n):
        if abs(b[i]) < abs(a[i]) + abs(c[i]):
            print(f"внимание: в строке {i + 1} условие устойчивости не выполнено")

    p = [0.0] * n
    q = [0.0] * n
    x = [0.0] * n

    # прямой ход
    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

# x_i = P_i*x_i+1 + Q_i
    for i in range(1, n):
        m = b[i] + a[i] * p[i - 1]
        p[i] = -c[i] / m
        q[i] = (d[i] - a[i] * q[i - 1]) / m

    # обратный ход
    x[n - 1] = q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    return x


# 17 вариант
a = [0, -1, -9, -1, 9]
b = [-6, 13, -15, -7, -18]
c = [5, 6, -4, 1, 0]
d = [51, 100, -12, 47, -90]

ans = solve_progonka(a, b, c, d)

for i, val in enumerate(ans):
    print(f"x{i + 1} = {val:.4f}")